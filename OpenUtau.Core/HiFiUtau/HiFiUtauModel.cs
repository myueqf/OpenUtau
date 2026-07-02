using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using K4os.Hash.xxHash;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Core.Util;

namespace OpenUtau.Core.HiFiUtau {
    class HiFiUtauModel {
        const int MaxFeatureCacheEntries = 512;

        public readonly string Location;
        public readonly HiFiUtauConfig Config;
        public readonly ulong Hash;

        readonly InferenceSession part1;
        readonly InferenceSession part2;
        readonly object featureCacheLock = new object();
        readonly Dictionary<ulong, float[,,]> featureCache = new Dictionary<ulong, float[,,]>();
        readonly Queue<ulong> featureCacheOrder = new Queue<ulong>();

        public HiFiUtauModel(string location) {
            Location = location;
            var part1Path = Path.Combine(location, "part1.onnx");
            var part2Path = Path.Combine(location, "part2.onnx");
            var configPath = Path.Combine(location, "config.json");
            if (!File.Exists(part1Path) || !File.Exists(part2Path) || !File.Exists(configPath)) {
                throw new MessageCustomizableException(
                    $"Invalid HiFiUTAU model folder \"{location}\"",
                    $"Invalid HiFiUTAU model folder: {location}",
                    new FileNotFoundException("HiFiUTAU model folder must contain part1.onnx, part2.onnx and config.json."));
            }

            Config = HiFiUtauConfig.Load(configPath);
            part1 = Onnx.getInferenceSession(part1Path);
            part2 = Onnx.getInferenceSession(part2Path);
            Hash = XXH64.DigestOf(File.ReadAllBytes(part1Path)) ^
                XXH64.DigestOf(File.ReadAllBytes(part2Path)) ^
                XXH64.DigestOf(File.ReadAllBytes(configPath));
        }

        public float[,,] ProcessFeatureSplice(HiFiUtauPhone[] phones) {
            double ratio = Config.FeatureHop / (double)Config.ModelHop;
            var segments = new List<float[,,]>();
            int previousEndFrame = 0;
            foreach (var phone in phones) {
                var feat = EncodeOne(phone.Mel, ratio, phone.ModelFrames > 0 ? phone.ModelFrames : null);
                if (feat == null) {
                    continue;
                }

                int gapFrames = Math.Max(0, phone.ModelStartFrame - previousEndFrame);
                if (gapFrames > 0) {
                    var gap = EncodeBlank(gapFrames, ratio);
                    if (gap != null) {
                        segments.Add(gap);
                    }
                }

                int overlapFrames = Math.Max(0, previousEndFrame - phone.ModelStartFrame);
                previousEndFrame = Math.Max(previousEndFrame, phone.ModelEndFrame);
                if (segments.Count == 0) {
                    segments.Add(feat);
                    continue;
                }

                int overlapFeat = Math.Min(
                    overlapFrames * Config.FeatUpsample,
                    Math.Min(segments[^1].GetLength(2), feat.GetLength(2)));
                if (overlapFeat <= 0) {
                    segments.Add(feat);
                    continue;
                }

                var last = segments[^1];
                segments[^1] = SliceFeat(last, 0, last.GetLength(2) - overlapFeat);
                segments.Add(CrossFadeFeat(
                    SliceFeat(last, last.GetLength(2) - overlapFeat, last.GetLength(2)),
                    SliceFeat(feat, 0, overlapFeat)));
                segments.Add(SliceFeat(feat, overlapFeat, feat.GetLength(2)));
            }
            return ConcatFeat(segments, Config.NumMels);
        }

        public float[] Synthesize(float[,,] feat, float[] f0Input) {
            int melFramesNeeded = feat.GetLength(2) / Config.FeatUpsample;
            var f0 = new float[melFramesNeeded];
            for (int i = 0; i < f0.Length; i++) {
                f0[i] = f0Input.Length == 0 ? 440 : f0Input[Math.Min(i, f0Input.Length - 1)];
            }
            if (Config.FrontPadFrames > 0) {
                feat = PadFeatEdges(feat, Config.FrontPadFrames * Config.FeatUpsample, left: true);
                f0 = PadF0(f0, Config.FrontPadFrames, left: true);
            }
            if (Config.TailPadFrames > 0) {
                feat = PadFeatEdges(feat, Config.TailPadFrames * Config.FeatUpsample, left: false);
                f0 = PadF0(f0, Config.TailPadFrames, left: false);
            }

            var wav = Part2Synthesize(feat, f0);
            int frontTrim = Config.FrontPadFrames * Config.ModelHop;
            int tailTrim = Config.TailPadFrames * Config.ModelHop;
            if (wav.Length > frontTrim) {
                wav = wav.Skip(frontTrim).ToArray();
            }
            if (wav.Length > tailTrim) {
                wav = wav.Take(wav.Length - tailTrim).ToArray();
            }
            return HiFiUtauMath.Resample(wav, Config.SampleRate, HiFiUtauConfig.OutputSampleRate);
        }

        float[,,]? EncodeOne(float[,]? mel, double ratio, int? targetModelFrames = null) {
            if (mel == null || mel.GetLength(1) == 0) {
                return null;
            }
            int targetFrames = targetModelFrames ??
                Math.Max(1, (int)Math.Round(mel.GetLength(1) * ratio, MidpointRounding.AwayFromZero));
            var melForEncoder = ResampleMel(mel, targetFrames);
            if (Config.EncoderPadFrames > 0) {
                melForEncoder = PadMelLeftRepeat(melForEncoder, Config.EncoderPadFrames);
            }

            ulong cacheKey = HashMel(melForEncoder);
            lock (featureCacheLock) {
                if (featureCache.TryGetValue(cacheKey, out var cached)) {
                    return cached;
                }
            }

            var feat = Part1Encode(melForEncoder);
            if (Config.EncoderPadFrames > 0) {
                feat = SliceFeat(feat, Config.EncoderPadFrames * Config.FeatUpsample, feat.GetLength(2));
            }
            lock (featureCacheLock) {
                if (!featureCache.ContainsKey(cacheKey)) {
                    featureCache[cacheKey] = feat;
                    featureCacheOrder.Enqueue(cacheKey);
                    while (featureCacheOrder.Count > MaxFeatureCacheEntries) {
                        featureCache.Remove(featureCacheOrder.Dequeue());
                    }
                }
            }
            return feat;
        }

        float[,,]? EncodeBlank(int modelFrames, double ratio) {
            if (modelFrames <= 0) {
                return null;
            }
            int featureFrames = Math.Max(1, (int)Math.Ceiling(modelFrames / ratio));
            return EncodeOne(HiFiUtauMath.CreateBlankMel(Config.NumMels, featureFrames), ratio, modelFrames);
        }

        float[,,] Part1Encode(float[,] mel) {
            var tensor = new DenseTensor<float>(FlattenMel(mel), new[] { 1, Config.NumMels, mel.GetLength(1) });
            using var results = part1.Run(new[] { NamedOnnxValue.CreateFromTensor("mel", tensor) });
            var output = results.First().AsTensor<float>();
            var dims = output.Dimensions.ToArray();
            return To3D(output.ToArray(), dims[0], dims[1], dims[2]);
        }

        float[] Part2Synthesize(float[,,] feat, float[] f0) {
            var featTensor = new DenseTensor<float>(
                FlattenFeat(feat),
                new[] { 1, feat.GetLength(1), feat.GetLength(2) });
            var f0Tensor = new DenseTensor<float>(f0, new[] { 1, f0.Length });
            using var results = part2.Run(new[] {
                NamedOnnxValue.CreateFromTensor("feat", featTensor),
                NamedOnnxValue.CreateFromTensor("f0", f0Tensor),
            });
            return results.First().AsTensor<float>().ToArray();
        }

        static float[] FlattenMel(float[,] mel) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var data = new float[bins * frames];
            for (int b = 0; b < bins; b++) {
                for (int t = 0; t < frames; t++) {
                    data[b * frames + t] = mel[b, t];
                }
            }
            return data;
        }

        static ulong HashMel(float[,] mel) {
            using var stream = new MemoryStream();
            using (var writer = new BinaryWriter(stream)) {
                int bins = mel.GetLength(0);
                int frames = mel.GetLength(1);
                writer.Write(bins);
                writer.Write(frames);
                for (int b = 0; b < bins; b++) {
                    for (int t = 0; t < frames; t++) {
                        writer.Write(BitConverter.SingleToUInt32Bits(mel[b, t]));
                    }
                }
            }
            return XXH64.DigestOf(stream.ToArray());
        }

        static float[] FlattenFeat(float[,,] feat) {
            int channels = feat.GetLength(1);
            int frames = feat.GetLength(2);
            var data = new float[channels * frames];
            for (int c = 0; c < channels; c++) {
                for (int t = 0; t < frames; t++) {
                    data[c * frames + t] = feat[0, c, t];
                }
            }
            return data;
        }

        static float[,,] To3D(float[] data, int n, int channels, int frames) {
            var result = new float[n, channels, frames];
            for (int c = 0; c < channels; c++) {
                for (int t = 0; t < frames; t++) {
                    result[0, c, t] = data[c * frames + t];
                }
            }
            return result;
        }

        static float[,] ResampleMel(float[,] mel, int targetFrames) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var result = new float[bins, targetFrames];
            for (int t = 0; t < targetFrames; t++) {
                double src = targetFrames == 1 ? 0 : t * (frames - 1.0) / (targetFrames - 1);
                int t0 = (int)Math.Floor(src);
                int t1 = Math.Min(frames - 1, t0 + 1);
                float f = (float)(src - t0);
                for (int b = 0; b < bins; b++) {
                    result[b, t] = mel[b, t0] + (mel[b, t1] - mel[b, t0]) * f;
                }
            }
            return result;
        }

        static float[,] PadMelLeftRepeat(float[,] mel, int count) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var result = new float[bins, count + frames];
            for (int b = 0; b < bins; b++) {
                for (int t = 0; t < count; t++) {
                    result[b, t] = mel[b, 0];
                }
                for (int t = 0; t < frames; t++) {
                    result[b, count + t] = mel[b, t];
                }
            }
            return result;
        }

        static float[,,] SliceFeat(float[,,] feat, int start, int end) {
            int channels = feat.GetLength(1);
            int frames = Math.Max(0, end - start);
            var result = new float[1, channels, frames];
            for (int c = 0; c < channels; c++) {
                for (int t = 0; t < frames; t++) {
                    result[0, c, t] = feat[0, c, start + t];
                }
            }
            return result;
        }

        static float[,,] CrossFadeFeat(float[,,] a, float[,,] b) {
            int channels = a.GetLength(1);
            int frames = Math.Min(a.GetLength(2), b.GetLength(2));
            var result = new float[1, channels, frames];
            for (int t = 0; t < frames; t++) {
                float fi = frames == 1 ? 1 : t / (float)(frames - 1);
                float fo = 1 - fi;
                for (int c = 0; c < channels; c++) {
                    result[0, c, t] = a[0, c, t] * fo + b[0, c, t] * fi;
                }
            }
            return result;
        }

        static float[,,] ConcatFeat(List<float[,,]> segments, int defaultChannels) {
            segments = segments.Where(s => s.GetLength(2) > 0).ToList();
            if (segments.Count == 0) {
                return new float[1, defaultChannels, 0];
            }
            int channels = segments[0].GetLength(1);
            int frames = segments.Sum(s => s.GetLength(2));
            var result = new float[1, channels, frames];
            int offset = 0;
            foreach (var segment in segments) {
                for (int c = 0; c < channels; c++) {
                    for (int t = 0; t < segment.GetLength(2); t++) {
                        result[0, c, offset + t] = segment[0, c, t];
                    }
                }
                offset += segment.GetLength(2);
            }
            return result;
        }

        static float[,,] PadFeatEdges(float[,,] feat, int count, bool left) {
            int channels = feat.GetLength(1);
            int frames = feat.GetLength(2);
            var result = new float[1, channels, frames + count];
            for (int c = 0; c < channels; c++) {
                for (int t = 0; t < result.GetLength(2); t++) {
                    int src = left ? Math.Max(0, t - count) : Math.Min(frames - 1, t);
                    result[0, c, t] = feat[0, c, src];
                }
            }
            return result;
        }

        static float[] PadF0(float[] f0, int count, bool left) {
            var result = new float[f0.Length + count];
            for (int i = 0; i < result.Length; i++) {
                int src = left ? Math.Max(0, i - count) : Math.Min(f0.Length - 1, i);
                result[i] = f0[src];
            }
            return result;
        }
    }
}

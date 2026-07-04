using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using OpenUtau.Classic;
using OpenUtau.Core.Format;
using OpenUtau.Core.Render;
using OpenUtau.Core.SignalChain;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Util;

namespace OpenUtau.Core.HiFiUtau {
    public class HiFiUtauRenderer : IRenderer {
        const int DynamicInterval = 5;
        static readonly Dictionary<string, HiFiUtauModel> models = new Dictionary<string, HiFiUtauModel>();
        static readonly object modelsLock = new object();

        static readonly HashSet<string> supportedExp = new HashSet<string>() {
            Format.Ustx.DYN,
            Format.Ustx.PITD,
            Format.Ustx.CLR,
            Format.Ustx.VEL,
            Format.Ustx.VOL,
            Format.Ustx.MOD,
            Format.Ustx.SHFT,
            Format.Ustx.GENC,
            Format.Ustx.BREC,
            Format.Ustx.TENC,
            Format.Ustx.VOIC,
            Format.Ustx.NORM,
            "phtp",
            "strt",
        };

        public USingerType SingerType => USingerType.Classic;
        public bool SupportsRenderPitch => false;

        public bool SupportsExpression(UExpressionDescriptor descriptor) {
            return supportedExp.Contains(descriptor.abbr);
        }

        public RenderResult Layout(RenderPhrase phrase) {
            return new RenderResult() {
                leadingMs = phrase.leadingMs,
                positionMs = phrase.positionMs,
                estimatedLengthMs = phrase.durationMs + phrase.leadingMs,
            };
        }

        public Task<RenderResult> Render(RenderPhrase phrase, Progress progress, int trackNo, CancellationTokenSource cancellation, bool isPreRender = false) {
            return Task.Run(() => {
                var result = Layout(phrase);
                try {
                    string progressInfo = $"Track {trackNo + 1}: {this} \"{string.Join(" ", phrase.phones.Select(p => p.phoneme))}\"";
                    progress.Complete(0, progressInfo);

                    var modelPath = ResolveModelPath(Renderers.HIFIUTAU_DEFAULT_MODEL);
                    if (string.IsNullOrEmpty(modelPath)) {
                        throw new MessageCustomizableException(
                            "HiFiUTAU model package or folder is not set.",
                            "HiFiUTAU model package or folder is not set.",
                            new Exception("HiFiUTAU model package or folder is not set."));
                    }

                    if (cancellation.IsCancellationRequested) {
                        return result;
                    }

                    var model = GetModel(modelPath);
                    var finalWavPath = Path.Join(PathManager.Inst.CachePath, $"hifiutau-{model.Hash:x16}-{phrase.hash:x16}.wav");
                    var rawWavPath = Path.Join(PathManager.Inst.CachePath, $"hifiutau-raw-{model.Hash:x16}-{phrase.hash:x16}.wav");
                    phrase.AddCacheFile(finalWavPath);
                    phrase.AddCacheFile(rawWavPath);

                    if (File.Exists(finalWavPath)) {
                        using var waveStream = Wave.OpenFile(finalWavPath);
                        result.samples = Wave.GetSamples(waveStream.ToSampleProvider().ToMono(1, 0));
                    }
                    if (result.samples == null) {
                        if (File.Exists(rawWavPath)) {
                            using var waveStream = Wave.OpenFile(rawWavPath);
                            result.samples = Wave.GetSamples(waveStream.ToSampleProvider().ToMono(1, 0));
                        }
                        if (result.samples == null) {
                            var phones = HiFiUtauPhone.CreateAll(phrase);
                            result.samples = RenderFeaturePipeline(phones, phrase, model, cancellation.Token);
                            if (cancellation.IsCancellationRequested) {
                                return result;
                            }
                            HiFiUtauMath.ApplyPhraseEdgeEnvelope(phones, result.samples, HiFiUtauConfig.OutputSampleRate);
                            WriteCacheWave(rawWavPath, result.samples);
                        }
                        if (result.samples != null) {
                            AudioPostProcessor.Apply(phrase, result);
                            Renderers.ApplyDynamics(phrase, result);
                            WriteCacheWave(finalWavPath, result.samples);
                        }
                    }
                    progress.Complete(phrase.phones.Length, progressInfo);
                    return result;
                } catch (OperationCanceledException) when (cancellation.IsCancellationRequested) {
                    return result;
                }
            });
        }

        static HiFiUtauModel GetModel(string modelPath) {
            modelPath = Path.GetFullPath(modelPath);
            lock (modelsLock) {
                if (!models.TryGetValue(modelPath, out var model)) {
                    model = new HiFiUtauModel(modelPath);
                    models[modelPath] = model;
                }
                return model;
            }
        }

        static string ResolveModelPath(string modelPath) {
            modelPath = string.IsNullOrWhiteSpace(modelPath)
                ? Renderers.HIFIUTAU_DEFAULT_MODEL
                : modelPath.Trim();
            if (TryFindModelFolder(modelPath, out var directPath)) {
                return directPath;
            }
            var packagePath = PackageManager.Inst.GetInstalledPath(modelPath);
            if (!string.IsNullOrEmpty(packagePath) && TryFindModelFolder(packagePath, out var packageModelPath)) {
                return packageModelPath;
            }
            return modelPath;
        }

        static bool TryFindModelFolder(string location, out string modelPath) {
            modelPath = string.Empty;
            if (!Directory.Exists(location)) {
                return false;
            }
            if (IsModelFolder(location)) {
                modelPath = Path.GetFullPath(location);
                return true;
            }
            foreach (var dir in Directory.GetDirectories(location)) {
                if (IsModelFolder(dir)) {
                    modelPath = Path.GetFullPath(dir);
                    return true;
                }
            }
            return false;
        }

        static bool IsModelFolder(string location) {
            return File.Exists(Path.Combine(location, "part1.onnx")) &&
                File.Exists(Path.Combine(location, "part2.onnx")) &&
                File.Exists(Path.Combine(location, "config.json"));
        }

        static void WriteCacheWave(string path, float[] samples) {
            var source = new WaveSource(0, 0, 0, 1);
            source.SetSamples(samples);
            WaveFileWriter.CreateWaveFile16(path, new ExportAdapter(source).ToMono(1, 0));
        }

        float[] RenderFeaturePipeline(
            HiFiUtauPhone[] phones,
            RenderPhrase phrase,
            HiFiUtauModel model,
            CancellationToken cancellation) {
            var melExtractor = new HiFiUtauMelExtractor(model.Config);
            foreach (var phone in phones) {
                if (cancellation.IsCancellationRequested) {
                    return Array.Empty<float>();
                }
                phone.Mel = ExtractFeatureMel(phone, model.Config, melExtractor);
                phone.Gender = SamplePhoneGender(phrase, phone, model.Config);
                ApplyPerPhoneControls(phone);
            }
            AlignPhoneModelFrames(phones, phrase, model.Config);
            MatchPhtp(phones, model.Config.MsPerFeatureFrame);
            var f0 = SampleF0(phrase, model.Config.ModelHop, model.Config.SampleRate);
            var feat = model.ProcessFeatureSplice(phones);
            return model.Synthesize(feat, f0);
        }

        float[,] ExtractFeatureMel(HiFiUtauPhone phone, HiFiUtauConfig config, HiFiUtauMelExtractor melExtractor) {
            var audio = HiFiUtauMath.ReadMonoSamples(phone.AudioPath, config.SampleRate);
            double totalLenMs = audio.Length * 1000.0 / config.SampleRate;
            int startSample = HiFiUtauMath.ClampSample(phone.OffsetMs, config.SampleRate, audio.Length);
            int consonantSample = HiFiUtauMath.ClampSample(phone.OffsetMs + phone.ConsonantMs, config.SampleRate, audio.Length);
            int endSample = phone.CutoffMs > 0
                ? HiFiUtauMath.ClampSample(totalLenMs - phone.CutoffMs, config.SampleRate, audio.Length)
                : HiFiUtauMath.ClampSample(phone.OffsetMs + Math.Abs(phone.CutoffMs), config.SampleRate, audio.Length);
            if (endSample < consonantSample) {
                endSample = consonantSample;
            }

            double stretch = HiFiUtauMath.StretchFactor(phone.Velocity);
            double preToLeftMs = phone.PreutterMs * stretch + phone.Envelope[0].X;
            if (preToLeftMs < 0) {
                // 尽可能保留起点前的真实音频
                startSample = Math.Max(0, startSample - 12 * config.FeatureHop);
            }

            // 提取 mel 时加 STFT 上下文，再裁掉边缘补帧
            int padContext = (config.FftSize / 2 + config.FeatureHop - 1) / config.FeatureHop * config.FeatureHop;
            int padFront = Math.Min(padContext, startSample);
            int padTail = Math.Min(padContext, audio.Length - endSample);
            var centers = Enumerable.Range(0, Math.Max(0, (endSample - startSample + padFront + padTail - 1) / config.FeatureHop + 1))
                .Select(i => startSample - padFront + i * config.FeatureHop)
                .ToArray();
            var melExt = melExtractor.Extract(audio, centers);
            int cropFront = padFront / config.FeatureHop;
            int cropTail = padTail / config.FeatureHop;
            var melFull = HiFiUtauMath.SliceMel(melExt, cropFront, Math.Max(cropFront, melExt.GetLength(1) - cropTail));
            int nFrames = melFull.GetLength(1);
            if (nFrames == 0) {
                return new float[config.NumMels, 0];
            }

            int conSamples = Math.Max(0, consonantSample - startSample);
            int conFramesOrig = conSamples > 0 ? Math.Min(nFrames, Math.Max(1, (conSamples - config.FeatureHop) / config.FeatureHop + 1)) : 0;
            int vowFramesOrig = nFrames - conFramesOrig;
            double totalBudgetMs = phone.Envelope[4].X + phone.PreutterMs * stretch;
            int totalFrames = Math.Max(1, (int)(totalBudgetMs / config.MsPerFeatureFrame));
            int targetConFrames = Math.Max(1, (int)(conFramesOrig * stretch));
            targetConFrames = Math.Min(targetConFrames, Math.Max(1, totalFrames - 1));
            int targetVowFrames = Math.Max(0, totalFrames - targetConFrames);

            if (phone.StretchMode == 1 && vowFramesOrig > 1 && targetVowFrames > vowFramesOrig * 1.5) {
                melFull = HiFiUtauMath.ReflectPadVowel(
                    melFull,
                    conFramesOrig,
                    targetVowFrames - vowFramesOrig + Math.Min(4, vowFramesOrig / 2));
                nFrames = melFull.GetLength(1);
                vowFramesOrig = nFrames - conFramesOrig;
            }

            var melOut = HiFiUtauMath.ResamplePhoneMel(
                melFull,
                totalFrames,
                conFramesOrig,
                targetConFrames,
                vowFramesOrig,
                stretch);
            if (preToLeftMs > 0) {
                int leftCutFrames = (int)(preToLeftMs / config.MsPerFeatureFrame);
                melOut = HiFiUtauMath.SliceMel(melOut, Math.Min(leftCutFrames, melOut.GetLength(1)), melOut.GetLength(1));
            } else if (preToLeftMs < 0) {
                int leftPadFrames = (int)(-preToLeftMs / config.MsPerFeatureFrame);
                melOut = HiFiUtauMath.PadBlankLeft(melOut, leftPadFrames);
            }
            return melOut;
        }

        static void AlignPhoneModelFrames(HiFiUtauPhone[] phones, RenderPhrase phrase, HiFiUtauConfig config) {
            double phraseStartMs = phrase.positionMs - phrase.leadingMs;
            foreach (var phone in phones) {
                double startMs = phone.PositionMs + phone.Envelope[0].X - phraseStartMs;
                double endMs = phone.PositionMs + phone.Envelope[4].X - phraseStartMs;
                phone.ModelStartFrame = Math.Max(0, HiFiUtauMath.FramesForMs(startMs, config.MsPerModelFrame));
                phone.ModelEndFrame = Math.Max(phone.ModelStartFrame + 1, HiFiUtauMath.FramesForMs(endMs, config.MsPerModelFrame));
                phone.ModelFrames = phone.ModelEndFrame - phone.ModelStartFrame;
            }
        }

        static void ApplyPerPhoneControls(HiFiUtauPhone phone) {
            if (phone.Mel == null || phone.Mel.GetLength(1) == 0) {
                return;
            }
            if (Math.Abs(phone.Volume - 1.0) > 1e-6) {
                HiFiUtauMath.AddLogGain(phone.Mel, Math.Log(phone.Volume));
            }
            var normalize = phone.Normalize / 100.0;
            if (normalize > 0) {
                double rms = HiFiUtauMath.MelRms(phone.Mel);
                if (rms > 1e-12) {
                    double target = rms * (1 - normalize) + 0.5 * normalize;
                    HiFiUtauMath.AddLogGain(phone.Mel, Math.Log(target / rms));
                }
            }
            if (phone.Gender != null && phone.Gender.Any(value => Math.Abs(value) > 0.001f)) {
                HiFiUtauMath.WarpMelFrequency(phone.Mel, phone.Gender.Select(value => -value / 100f).ToArray());
            } else {
                double semitones = phone.ToneShift / 100.0;
                if (Math.Abs(semitones) > 0.001) {
                    HiFiUtauMath.WarpMelFrequency(phone.Mel, Math.Pow(2.0, semitones / 12.0));
                }
            }
        }

        static float[]? SamplePhoneGender(RenderPhrase phrase, HiFiUtauPhone phone, HiFiUtauConfig config) {
            if (phrase.gender == null || phrase.gender.Length == 0 || phone.Mel == null) {
                return null;
            }
            int frames = phone.Mel.GetLength(1);
            if (frames == 0) {
                return null;
            }
            var gender = new float[frames];
            double phoneStartMs = phone.PositionMs + phone.Envelope[0].X;
            for (int i = 0; i < frames; i++) {
                double posMs = phoneStartMs + i * config.MsPerFeatureFrame;
                int ticks = phrase.timeAxis.MsPosToTickPos(posMs) - (phrase.position - phrase.leading);
                int idx = Math.Clamp(ticks / DynamicInterval, 0, phrase.gender.Length - 1);
                gender[i] = phrase.gender[idx];
            }
            return gender;
        }

        static void MatchPhtp(HiFiUtauPhone[] phones, double msPerFrame) {
            for (int i = 0; i < phones.Length; i++) {
                var phone = phones[i];
                if (phone.Mel == null || phone.Mel.GetLength(1) == 0 || phone.PhonemeType == 0) {
                    continue;
                }
                if (phone.PhonemeType == 1 && i < phones.Length - 1) {
                    MatchEnergy(phone, phones[i + 1], msPerFrame, followNext: true);
                } else if (phone.PhonemeType == 2 && i > 0) {
                    MatchEnergy(phone, phones[i - 1], msPerFrame, followNext: false);
                }
            }
        }

        static void MatchEnergy(HiFiUtauPhone target, HiFiUtauPhone reference, double msPerFrame, bool followNext) {
            if (target.Mel == null || reference.Mel == null) {
                return;
            }
            var env = followNext ? reference.Envelope : target.Envelope;
            double overlapMs = env[1].X < 0 ? Math.Abs(env[0].X) - Math.Abs(env[1].X) : Math.Abs(env[1].X) + Math.Abs(env[0].X);
            int frames = (int)Math.Round(overlapMs / msPerFrame);
            int targetFrames = Math.Min(frames, target.Mel.GetLength(1));
            int refFrames = Math.Min(frames, reference.Mel.GetLength(1));
            if (targetFrames <= 0 || refFrames <= 0) {
                return;
            }
            double targetRms = HiFiUtauMath.MelRms(target.Mel, followNext ? target.Mel.GetLength(1) - targetFrames : 0, targetFrames);
            double refRms = HiFiUtauMath.MelRms(reference.Mel, followNext ? 0 : reference.Mel.GetLength(1) - refFrames, refFrames);
            if (targetRms > 1e-12 && refRms > 1e-12) {
                HiFiUtauMath.AddLogGain(target.Mel, Math.Log(refRms / targetRms));
            }
        }

        float[] SampleF0(RenderPhrase phrase, int targetHop, int sampleRate) {
            int frames = Math.Max(1, (int)Math.Ceiling((phrase.durationMs + phrase.leadingMs) * sampleRate / 1000.0 / targetHop));
            var f0 = new float[frames];
            for (int i = 0; i < frames; i++) {
                double posMs = phrase.positionMs - phrase.leadingMs + i * targetHop * 1000.0 / sampleRate;
                int ticks = phrase.timeAxis.MsPosToTickPos(posMs) - (phrase.position - phrase.leading);
                int idx = Math.Clamp(ticks / DynamicInterval, 0, phrase.pitches.Length - 1);
                f0[i] = (float)MusicMath.ToneToFreq(phrase.pitches[idx] * 0.01);
            }
            return f0;
        }

        public RenderPitchResult LoadRenderedPitch(RenderPhrase phrase) {
            return null;
        }

        public List<RenderRealCurveResult> LoadRenderedRealCurves(RenderPhrase phrase) {
            return new List<RenderRealCurveResult>(0);
        }

        public UExpressionDescriptor[] GetSuggestedExpressions(USinger singer, URenderSettings renderSettings) {
            return new[] {
                new UExpressionDescriptor("phoneme type", "phtp", false, new[] { "normal", "follow next", "follow previous" }),
                new UExpressionDescriptor("stretch mode", "strt", false, new[] { "stretch", "loop" }),
            };
        }

        public override string ToString() => Renderers.HIFIUTAU;
    }
}

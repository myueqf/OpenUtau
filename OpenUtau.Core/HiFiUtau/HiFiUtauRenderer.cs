using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using K4os.Hash.xxHash;
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
            Format.Ustx.ATK,
            Format.Ustx.DEC,
            Format.Ustx.MODP,
            Format.Ustx.GENC,
            Format.Ustx.BREC,
            Format.Ustx.TENC,
            Format.Ustx.VOIC,
            Format.Ustx.NORM,
            "phtp",
            "stm",
            "brel",
            "breh",
            "bric",
            "gwlc",
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
                            "HiFiUTAU model package or folder is not set. Please download from: https://github.com/xiaobaijunya/HIFIUTAU_model/releases",
                            "HiFiUTAU model package or folder is not set. Please download from: https://github.com/xiaobaijunya/HIFIUTAU_model/releases",
                            new Exception("HiFiUTAU model package or folder is not set."));
                    }

                    if (cancellation.IsCancellationRequested) {
                        return result;
                    }

                    var model = GetModel(modelPath);

                    // New cache directory structure
                    var cacheDir = Path.Join(PathManager.Inst.CachePath, "hifiutau");
                    var rawDir = Path.Join(cacheDir, "raw");
                    var hnsepDir = Path.Join(cacheDir, "hnsep");
                    var finalDir = Path.Join(cacheDir, "final");
                    Directory.CreateDirectory(rawDir);
                    Directory.CreateDirectory(hnsepDir);
                    Directory.CreateDirectory(finalDir);

                    var rawHash = ComputeRawHash(phrase);
                    var rawWavPath = Path.Join(rawDir, $"{model.Hash:x16}-{rawHash:x16}.wav");
                    var finalWavPath = Path.Join(finalDir, $"{model.Hash:x16}-{phrase.hash:x16}.wav");
                    var hnsepHarmonicPath = Path.Join(hnsepDir, $"harmonic-{model.Hash:x16}-{rawHash:x16}.wav");
                    var hnsepNoisePath = Path.Join(hnsepDir, $"noise-{model.Hash:x16}-{rawHash:x16}.wav");
                    phrase.AddCacheFile(finalWavPath);
                    phrase.AddCacheFile(rawWavPath);
                    phrase.AddCacheFile(hnsepHarmonicPath);
                    phrase.AddCacheFile(hnsepNoisePath);

                    if (File.Exists(finalWavPath)) {
                        result.samples = LoadCacheWave(finalWavPath);
                    }
                    if (result.samples == null) {
                        if (File.Exists(rawWavPath)) {
                            result.samples = LoadCacheWave(rawWavPath);
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
                            // HN-SEP processing with caching
                            var postCurves = PostProcessCurves.FromPhrase(phrase);
                            if (postCurves.NeedsHnsep) {
                                float[] harmonic, noise;
                                if (File.Exists(hnsepHarmonicPath) && File.Exists(hnsepNoisePath)) {
                                    harmonic = LoadCacheWave(hnsepHarmonicPath);
                                    noise = LoadCacheWave(hnsepNoisePath);
                                } else {
                                    var hnsep = AudioPostProcessor.GetSeparator();
                                    (harmonic, noise) = hnsep.Separate(result.samples);
                                    WriteCacheWave(hnsepHarmonicPath, harmonic);
                                    WriteCacheWave(hnsepNoisePath, noise);
                                }
                                AudioPostProcessor.ApplyWithSeparated(phrase, result, harmonic, noise,
                                    postCurves.Brel, postCurves.Breh, postCurves.Bri);
                            } else {
                                AudioPostProcessor.Apply(phrase, result);
                            }
                            Renderers.ApplyDynamics(phrase, result);
                            if (postCurves.NeedsGrowl) {
                                var pitchHzCurve = AudioPostProcessingDsp.PitchHzCurve(phrase, result.samples.Length);
                                AudioPostProcessor.ApplyGrowl(result.samples, postCurves.Growl, AudioPostProcessingDsp.SampleRate, pitchHzCurve);
                            }
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

        static ulong ComputeRawHash(RenderPhrase phrase) {
            using var stream = new MemoryStream();
            using (var writer = new BinaryWriter(stream)) {
                writer.Write(phrase.preEffectHash);
                WriteCurve(writer, phrase.pitches);
                WriteCurve(writer, phrase.gender);
                WriteCurve(writer, phrase.toneShift);
                WriteCurve(writer, GetCurve(phrase, "gwlc"));
                foreach (var phone in phrase.phones) {
                    writer.Write(phone.toneShift);
                }
            }
            return XXH64.DigestOf(stream.ToArray());
        }

        static float[]? GetCurve(RenderPhrase phrase, string abbr) {
            return phrase.curves.FirstOrDefault(c => c.Item1 == abbr)?.Item2;
        }

        static void WriteCurve(BinaryWriter writer, float[]? curve) {
            if (curve == null) {
                writer.Write("null");
                return;
            }
            writer.Write(curve.Length);
            foreach (var value in curve) {
                writer.Write(value);
            }
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

        static float[] LoadCacheWave(string path) {
            using var waveStream = Wave.OpenFile(path);
            return Wave.GetSamples(waveStream.ToSampleProvider().ToMono(1, 0));
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
            MatchPhtp(phones, model.Config.MsPerFeatureFrame);
            foreach (var phone in phones) {
                ApplyPhoneEnvelope(phone);
            }
            AlignPhoneModelFrames(phones, phrase, model.Config);
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

            if (phone.StretchMode == (int)StretchMode.Loop && vowFramesOrig > 1 && targetVowFrames > vowFramesOrig * 1.5) {
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

            // strt=1 (loop): normalize vowel energy to linear gradient
            if (phone.StretchMode == (int)StretchMode.Loop && targetVowFrames > 1 &&
                melOut.GetLength(1) >= targetConFrames + 2) {
                HiFiUtauMath.NormalizeLoopVowelEnergy(melOut, targetConFrames);
            }

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

        static void ApplyPhoneEnvelope(HiFiUtauPhone phone) {
            if (phone.Mel == null || phone.Mel.GetLength(1) == 0) {
                return;
            }
            // Apply per-phone envelope amplitude after phtp (replaces Volume parameter)
            if (phone.Envelope != null && phone.Envelope.Length >= 5) {
                HiFiUtauMath.ApplyEnvelopeToMel(phone.Mel, phone.Envelope);
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
                new UExpressionDescriptor("phoneme type", "phtp", true, new[] { "normal", "follow next", "follow previous" }),
                new UExpressionDescriptor("stretch mode", "stm", true, new[] { "none", "loop" }),
                new UExpressionDescriptor("modulation plus", Format.Ustx.MODP, 0, 100, 0),
                new UExpressionDescriptor {
                    name = "breath low (curve)",
                    abbr = "brel",
                    type = UExpressionType.Curve,
                    min = -100,
                    max = 100,
                    defaultValue = 0,
                    isFlag = false,
                },
                new UExpressionDescriptor {
                    name = "breath high (curve)",
                    abbr = "breh",
                    type = UExpressionType.Curve,
                    min = -100,
                    max = 100,
                    defaultValue = 0,
                    isFlag = false,
                },
                new UExpressionDescriptor {
                    name = "brightness (curve)",
                    abbr = "bric",
                    type = UExpressionType.Curve,
                    min = -100,
                    max = 100,
                    defaultValue = 0,
                    isFlag = false,
                },
                new UExpressionDescriptor {
                    name = "growl (curve)",
                    abbr = "gwlc",
                    type = UExpressionType.Curve,
                    min = 0,
                    max = 100,
                    defaultValue = 0,
                    isFlag = false,
                },
            };
        }

        public override string ToString() => Renderers.HIFIUTAU;

        readonly struct PostProcessCurves {
            PostProcessCurves(float[]? brel, float[]? breh, float[]? bri, float[]? growl, bool needsHnsep, bool needsGrowl) {
                Brel = brel;
                Breh = breh;
                Bri = bri;
                Growl = growl;
                NeedsHnsep = needsHnsep;
                NeedsGrowl = needsGrowl;
            }

            public readonly float[]? Brel;
            public readonly float[]? Breh;
            public readonly float[]? Bri;
            public readonly float[]? Growl;
            public readonly bool NeedsHnsep;
            public readonly bool NeedsGrowl;

            public static PostProcessCurves FromPhrase(RenderPhrase phrase) {
                var brel = GetCurve(phrase, "brel");
                var breh = GetCurve(phrase, "breh");
                var bri = GetCurve(phrase, "bric");
                var growl = GetCurve(phrase, "gwlc");
                bool needsHnsep =
                    AudioPostProcessor.HasNonDefaultCurve(phrase.breathiness, 0, 0.5f) ||
                    AudioPostProcessor.HasNonDefaultCurve(phrase.tension, 0, 0.5f) ||
                    AudioPostProcessor.HasNonDefaultCurve(phrase.voicing, 100, 0.5f) ||
                    AudioPostProcessor.HasNonDefaultCurve(brel, 0, 0.5f) ||
                    AudioPostProcessor.HasNonDefaultCurve(breh, 0, 0.5f) ||
                    AudioPostProcessor.HasNonDefaultCurve(bri, 0, 0.5f);
                bool needsGrowl = AudioPostProcessor.HasNonDefaultCurve(growl, 0, 0.5f);
                return new PostProcessCurves(brel, breh, bri, growl, needsHnsep, needsGrowl);
            }
        }
    }
}

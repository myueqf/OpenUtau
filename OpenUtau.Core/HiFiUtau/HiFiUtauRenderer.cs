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
            Format.Ustx.CLRY,
            Format.Ustx.XSY,
            Format.Ustx.VEL,
            Format.Ustx.VOL,
            Format.Ustx.ATK,
            Format.Ustx.DEC,
            Format.Ustx.MODP,
            Format.Ustx.SHFT,
            Format.Ustx.GENC,
            Format.Ustx.BREC,
            Format.Ustx.TENC,
            Format.Ustx.VOIC,
            Format.Ustx.NORM,
            Format.Ustx.DIR,
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

        public Task<RenderResult> Render(RenderPhrase phrase, Progress progress, int trackNo, CancellationTokenSource cancellation, bool isPreRender = false, RenderPhraseEvents? renderEvents = null) {
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
                    var phones = HiFiUtauPhone.CreateAll(phrase);
                    AlignPhoneModelFrames(phones, phrase, model.Config);
                    HiFiUtauPhone[]? secondaryPhones = null;
                    var controlPhones = phones;
                    if (phrase.secondaryPhones != null && phrase.xsy != null && phrase.xsy.Any(value => value > 0)) {
                        secondaryPhones = HiFiUtauPhone.CreateAll(phrase.secondaryPhones);
                        AlignPhoneModelFrames(secondaryPhones, phrase, model.Config);
                        controlPhones = phones.Select((phone, i) => phone.WithTiming(
                            secondaryPhones[i], SampleCrossSynthesisAt(phrase, phone.PositionMs))).ToArray();
                        AlignPhoneModelFrames(controlPhones, phrase, model.Config);
                    }

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
                    var finalWavPath = Path.Join(finalDir, $"{model.Hash:x16}-{rawHash:x16}-{phrase.hash:x16}-classic-direct-v1.wav");
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
                            result.samples = RenderFeaturePipeline(phones, secondaryPhones, phrase, model, cancellation.Token);
                            if (cancellation.IsCancellationRequested) {
                                return result;
                            }
                            ApplyPhraseEdges(phones, secondaryPhones, phrase, result.samples);
                            WriteCacheWave(rawWavPath, result.samples);
                        }
                        if (result.samples != null) {
                            // HN-SEP processing with caching
                            var postCurves = PostProcessCurves.FromPhrase(phrase, controlPhones);
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
                                    postCurves.Brel, postCurves.Breh, postCurves.Bri,
                                    postCurves.Breathiness, postCurves.Tension, postCurves.Voicing);
                            } else {
                                AudioPostProcessor.Apply(phrase, result,
                                    postCurves.Breathiness, postCurves.Tension, postCurves.Voicing);
                            }
                            if (postCurves.NeedsGrowl) {
                                var pitchHzCurve = AudioPostProcessingDsp.PitchHzCurve(phrase, result.samples.Length);
                                AudioPostProcessor.ApplyGrowl(result.samples, postCurves.Growl, AudioPostProcessingDsp.SampleRate, pitchHzCurve);
                            }
                            double samplesPerModelFrame =
                                model.Config.ModelHop * (double)HiFiUtauConfig.OutputSampleRate / model.Config.SampleRate;
                            HiFiUtauLoudnessNormalizer.NormalizePhonesInPlace(
                                result.samples,
                                controlPhones,
                                HiFiUtauConfig.OutputSampleRate,
                                samplesPerModelFrame);
                            // Overlay raw samples for direct phonemes. Runs after loudness
                            // normalization (so the raw recording is not re-leveled) and before
                            // ApplyPhoneVolumes (so the VOL expression also scales the direct audio).
                            ApplyDirectPhones(result.samples, phones, phrase, HiFiUtauConfig.OutputSampleRate);
                            // Apply VOL on the waveform so its percentage remains a linear output ratio.
                            ApplyPhoneVolumes(
                                result.samples,
                                controlPhones,
                                model.Config.ModelHop * (double)HiFiUtauConfig.OutputSampleRate / model.Config.SampleRate);
                            Renderers.ApplyDynamics(phrase, result);
                            WriteCacheWave(finalWavPath, result.samples);
                        }
                    }
                    progress.Complete(phrase.phones.Length, progressInfo);
                    if (result.samples != null) {
                        PlaybackManager.Inst.LiveWaveformCache[phrase.hash.ToString()] = (
                            trackNo, phrase.positionMs - phrase.leadingMs, result.samples, DateTime.Now);
                        Task.Factory.StartNew(() => {
                            DocManager.Inst.ExecuteCmd(new WaveformReadyNotification());
                        }, CancellationToken.None, TaskCreationOptions.None, DocManager.Inst.MainScheduler);
                    }
                    return result;
                } catch (OperationCanceledException) when (cancellation.IsCancellationRequested) {
                    return result;
                }
            });
        }

        static ulong ComputeRawHash(RenderPhrase phrase) {
            using var stream = new MemoryStream();
            using (var writer = new BinaryWriter(stream)) {
                writer.Write("hifiutau-v14-independent-cross-synthesis-timing");
                writer.Write(phrase.preEffectHash);
                WriteCurve(writer, phrase.pitches);
                WriteCurve(writer, phrase.xsy);
                WriteCurve(writer, phrase.gender);
                WriteCurveActivity(writer, phrase.genderCurveActive);
                WriteCurve(writer, phrase.toneShift);
                WriteCurve(writer, GetCurve(phrase, "gwlc"));
                foreach (var phone in phrase.phones) {
                    writer.Write(phone.gender);
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

        static void ApplyDirectPhones(float[] samples, HiFiUtauPhone[] phones, RenderPhrase phrase, int sampleRate) {
            if (samples == null || samples.Length == 0 || phones == null || phones.Length == 0 || sampleRate <= 0) {
                return;
            }

            double phraseStartMs = phrase.positionMs - phrase.leadingMs;
            foreach (var phone in phones) {
                if (!phone.Direct || string.IsNullOrEmpty(phone.AudioPath)) {
                    continue;
                }

                var source = HiFiUtauMath.ReadMonoSamples(phone.AudioPath, sampleRate);
                var directSamples = SliceDirectSamples(source, phone, sampleRate);
                ApplyDirectPhone(samples, directSamples, phone, phraseStartMs, sampleRate);
            }
        }

        internal static float[] SliceDirectSamples(float[] source, HiFiUtauPhone phone, int sampleRate) {
            if (source == null || source.Length == 0 || phone == null || sampleRate <= 0) {
                return Array.Empty<float>();
            }

            int offset = HiFiUtauMath.ClampSample(phone.OffsetMs, sampleRate, source.Length);
            double totalLengthMs = source.Length * 1000.0 / sampleRate;
            int end = phone.CutoffMs >= 0
                ? HiFiUtauMath.ClampSample(totalLengthMs - phone.CutoffMs, sampleRate, source.Length)
                : HiFiUtauMath.ClampSample(phone.OffsetMs + Math.Abs(phone.CutoffMs), sampleRate, source.Length);
            end = Math.Max(offset, end);
            var result = new float[end - offset];
            Array.Copy(source, offset, result, 0, result.Length);
            return result;
        }

        internal static void ApplyDirectPhone(
            float[] destination,
            float[] source,
            HiFiUtauPhone phone,
            double phraseStartMs,
            int sampleRate) {
            if (destination == null || destination.Length == 0 || source == null || source.Length == 0 ||
                phone == null || phone.Envelope == null || phone.Envelope.Length < 5 || sampleRate <= 0) {
                return;
            }

            double stretch = HiFiUtauMath.StretchFactor(phone.Velocity);
            int skipOverSamples = (int)((phone.PreutterMs * stretch - phone.LeadingMs) * sampleRate / 1000.0);
            int segmentStart = (int)Math.Round(
                (phone.PositionMs - phone.LeadingMs - phraseStartMs) * sampleRate / 1000.0,
                MidpointRounding.AwayFromZero);
            double envelopeShift = -phone.Envelope[0].X;
            double baseVolume = double.IsFinite(phone.Volume) ? Math.Max(0, phone.Volume) : 1.0;
            int nextPoint = 0;
            for (int sourceIndex = 0; sourceIndex < source.Length; sourceIndex++) {
                int destinationIndex = segmentStart + sourceIndex - skipOverSamples;
                if (destinationIndex < 0 || destinationIndex >= destination.Length) {
                    continue;
                }

                while (nextPoint < phone.Envelope.Length &&
                    sourceIndex > EnvelopeSample(phone.Envelope[nextPoint].X)) {
                    nextPoint++;
                }

                double gain;
                if (nextPoint == 0) {
                    gain = phone.Envelope[0].Y / 100.0;
                } else if (nextPoint >= phone.Envelope.Length) {
                    gain = phone.Envelope[^1].Y / 100.0;
                } else {
                    double x0 = EnvelopeSample(phone.Envelope[nextPoint - 1].X);
                    double x1 = EnvelopeSample(phone.Envelope[nextPoint].X);
                    double y0 = phone.Envelope[nextPoint - 1].Y / 100.0;
                    double y1 = phone.Envelope[nextPoint].Y / 100.0;
                    gain = x0 >= x1
                        ? y0
                        : y0 + (y1 - y0) * (sourceIndex - x0) / (x1 - x0);
                }

                // Classic mixes independently enveloped segments. Blend the direct segment into the
                // phrase-level model output so its fade-out does not erase the next consonant.
                double blend = baseVolume > 1e-9
                    ? Math.Clamp(gain / baseVolume, 0.0, 1.0)
                    : 0.0;
                destination[destinationIndex] =
                    destination[destinationIndex] * (float)(1.0 - blend) +
                    source[sourceIndex] * (float)gain;
            }

            double EnvelopeSample(double x) =>
                (x + envelopeShift) * sampleRate / 1000.0 + skipOverSamples;
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
            HiFiUtauPhone[]? secondaryPhones,
            RenderPhrase phrase,
            HiFiUtauModel model,
            CancellationToken cancellation) {
            var melExtractor = new HiFiUtauMelExtractor(model.Config);
            PreparePhoneMels(phones, phrase, model.Config, melExtractor, cancellation);
            int totalFrames = Math.Max(phones.Max(p => p.ModelEndFrame),
                secondaryPhones?.Max(p => p.ModelEndFrame) ?? 0);
            var feat = model.ProcessFeatureSplice(phones, totalFrames);
            if (secondaryPhones != null) {
                PreparePhoneMels(secondaryPhones, phrase, model.Config, melExtractor, cancellation);
                var secondaryFeat = model.ProcessFeatureSplice(secondaryPhones, totalFrames);
                var ratios = SampleCrossSynthesis(phrase, model.Config, feat.GetLength(2));
                HiFiUtauModel.BlendFeaturesInPlace(feat, secondaryFeat, ratios);
            }
            cancellation.ThrowIfCancellationRequested();
            var f0 = SampleF0(phrase, model.Config.ModelHop, model.Config.SampleRate);
            return model.Synthesize(feat, f0);
        }

        void PreparePhoneMels(
            HiFiUtauPhone[] phones,
            RenderPhrase phrase,
            HiFiUtauConfig config,
            HiFiUtauMelExtractor melExtractor,
            CancellationToken cancellation) {
            foreach (var phone in phones) {
                cancellation.ThrowIfCancellationRequested();
                phone.Mel = ExtractFeatureMel(phone, config, melExtractor);
                phone.Gender = SamplePhoneGender(phrase, phone, config);
                ApplyPerPhoneControls(phone);
            }
            MatchPhtp(phones, config.MsPerFeatureFrame);
            foreach (var phone in phones) {
                ApplyPhoneEnvelope(phone);
            }
        }

        static float[] SampleCrossSynthesis(RenderPhrase phrase, HiFiUtauConfig config, int frames) {
            var ratios = new float[frames];
            if (phrase.xsy == null || phrase.xsy.Length == 0) {
                return ratios;
            }
            double startMs = phrase.positionMs - phrase.leadingMs;
            double msPerFrame = config.MsPerModelFrame / config.FeatUpsample;
            for (int i = 0; i < frames; i++) {
                ratios[i] = SampleCrossSynthesisAt(phrase, startMs + i * msPerFrame);
            }
            return ratios;
        }

        static float SampleCrossSynthesisAt(RenderPhrase phrase, double positionMs) {
            double tick = phrase.timeAxis.MsPosToNonExactTickPos(positionMs);
            double index = Math.Clamp(
                (tick - (phrase.position - phrase.leading)) / UCurve.interval,
                0, phrase.xsy.Length - 1);
            int left = (int)index;
            int right = Math.Min(left + 1, phrase.xsy.Length - 1);
            float value = phrase.xsy[left] + (phrase.xsy[right] - phrase.xsy[left]) * (float)(index - left);
            return float.IsFinite(value) ? Math.Clamp(value / 100f, 0f, 1f) : 0f;
        }

        static void ApplyPhraseEdges(HiFiUtauPhone[] phones, HiFiUtauPhone[]? secondaryPhones,
            RenderPhrase phrase, float[] samples) {
            double startMs = phrase.positionMs - phrase.leadingMs;
            const int sampleRate = HiFiUtauConfig.OutputSampleRate;
            if (secondaryPhones == null) {
                HiFiUtauMath.ApplyPhraseEdgeEnvelope(phones, samples, sampleRate,
                    phrase.secondaryPhones == null ? null : startMs);
                return;
            }
            var primaryEnvelope = new float[samples.Length];
            var secondaryEnvelope = new float[samples.Length];
            Array.Fill(primaryEnvelope, 1f);
            Array.Fill(secondaryEnvelope, 1f);
            HiFiUtauMath.ApplyPhraseEdgeEnvelope(phones, primaryEnvelope, sampleRate, startMs);
            HiFiUtauMath.ApplyPhraseEdgeEnvelope(secondaryPhones, secondaryEnvelope, sampleRate, startMs);
            for (int i = 0; i < samples.Length; i++) {
                float gain = primaryEnvelope[i];
                if (gain != secondaryEnvelope[i]) {
                    float ratio = SampleCrossSynthesisAt(phrase, startMs + i * 1000.0 / sampleRate);
                    gain += (secondaryEnvelope[i] - gain) * ratio;
                }
                samples[i] *= gain;
            }
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
            // A fixed region may occupy the complete destination. Keep it
            // monotonic instead of reserving a fake one-frame vowel tail.
            targetConFrames = Math.Min(targetConFrames, totalFrames);
            var melOut = phone.StretchMode == (int)StretchMode.Loop
                ? HiFiUtauMath.ResamplePhoneMelLoop(
                    melFull, totalFrames, conFramesOrig, targetConFrames, vowFramesOrig, stretch)
                : HiFiUtauMath.ResamplePhoneMel(
                    melFull, totalFrames, conFramesOrig, targetConFrames, vowFramesOrig, stretch);

            if (preToLeftMs > 0) {
                int leftCutFrames = (int)(preToLeftMs / config.MsPerFeatureFrame);
                melOut = HiFiUtauMath.SliceMel(melOut, Math.Min(leftCutFrames, melOut.GetLength(1)), melOut.GetLength(1));
            } else if (preToLeftMs < 0) {
                int leftPadFrames = (int)(-preToLeftMs / config.MsPerFeatureFrame);
                melOut = phone.StretchMode == (int)StretchMode.Loop
                    ? HiFiUtauMath.PadBlankLeftFadeIn(melOut, leftPadFrames)
                    : HiFiUtauMath.PadBlankLeft(melOut, leftPadFrames);
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
            if (phone.Gender != null && phone.Gender.Any(value => Math.Abs(value) > 0.001f)) {
                HiFiUtauMath.WarpMelFrequency(phone.Mel, phone.Gender.Select(value => -value / 100f).ToArray());
            } else {
                double semitones = phone.ToneShift / 100.0;
                if (Math.Abs(semitones) > 0.001) {
                    HiFiUtauMath.WarpMelFrequency(phone.Mel, Math.Pow(2.0, semitones / 12.0));
                }
            }
        }

        internal static void ApplyPhoneVolumes(
            float[] samples,
            HiFiUtauPhone[] phones,
            double samplesPerModelFrame) {
            if (samples == null || samples.Length == 0 ||
                phones == null || phones.Length == 0 ||
                !double.IsFinite(samplesPerModelFrame) || samplesPerModelFrame <= 0) {
                return;
            }

            static float GetGain(HiFiUtauPhone phone) {
                return double.IsFinite(phone.Volume)
                    ? (float)Math.Max(0, phone.Volume)
                    : 1f;
            }

            int ToSample(int frame) => Math.Clamp(
                (int)Math.Round(frame * samplesPerModelFrame, MidpointRounding.AwayFromZero),
                0,
                samples.Length);

            var gains = new float[samples.Length];
            float previousGain = GetGain(phones[0]);
            Array.Fill(gains, previousGain);
            int previousEnd = ToSample(phones[0].ModelEndFrame);

            for (int i = 1; i < phones.Length; i++) {
                int start = ToSample(phones[i].ModelStartFrame);
                int end = ToSample(phones[i].ModelEndFrame);
                if (end <= start) {
                    continue;
                }

                float gain = GetGain(phones[i]);
                if (start < previousEnd) {
                    int overlapEnd = Math.Min(previousEnd, end);
                    int overlapSamples = overlapEnd - start;
                    float overlapStartGain = gains[start];
                    for (int j = start; j < overlapEnd; j++) {
                        float alpha = overlapSamples == 1
                            ? 1f
                            : (j - start) / (float)(overlapSamples - 1);
                        gains[j] = overlapStartGain + (gain - overlapStartGain) * alpha;
                    }
                    Array.Fill(gains, gain, overlapEnd, end - overlapEnd);
                } else {
                    Array.Fill(gains, previousGain, previousEnd, start - previousEnd);
                    Array.Fill(gains, gain, start, end - start);
                }

                if (end >= previousEnd) {
                    previousEnd = end;
                    previousGain = gain;
                }
            }

            Array.Fill(gains, previousGain, previousEnd, samples.Length - previousEnd);
            for (int i = 0; i < samples.Length; i++) {
                samples[i] *= gains[i];
            }
        }

        static void ApplyPhoneEnvelope(HiFiUtauPhone phone) {
            if (phone.Mel == null || phone.Mel.GetLength(1) == 0) {
                return;
            }
            // Apply the crossfade envelope after phtp. VOL is applied to the waveform later.
            if (phone.Envelope != null && phone.Envelope.Length >= 5) {
                HiFiUtauMath.ApplyEnvelopeToMel(phone.Mel, phone.Envelope);
            }
        }

        static float[]? SamplePhoneGender(RenderPhrase phrase, HiFiUtauPhone phone, HiFiUtauConfig config) {
            bool hasPhraseCurve = phrase.gender != null && phrase.gender.Length > 0;
            bool hasPhoneValue = Math.Abs(phone.GenderValue) > 0.001f;
            if ((!hasPhraseCurve && !hasPhoneValue) || phone.Mel == null) {
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
                float value = phone.GenderValue;
                if (hasPhraseCurve) {
                    int idx = Math.Clamp(ticks / DynamicInterval, 0, phrase.gender.Length - 1);
                    if (idx < phrase.genderCurveActive.Length && phrase.genderCurveActive[idx]) {
                        value = phrase.gender[idx];
                    }
                }
                gender[i] = Math.Clamp(value, -100, 100);
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
                new UExpressionDescriptor("phoneme type", "phtp", true, new[] { "normal", "follow next", "follow previous" }, skipOutputIfDefault: true),
                new UExpressionDescriptor("stretch mode", "stm", true, new[] { "none", "loop" }, skipOutputIfDefault: true),
                new UExpressionDescriptor("direct", Format.Ustx.DIR, false, new[] { "off", "on" }, skipOutputIfDefault: true),
                new UExpressionDescriptor("modulation plus", Format.Ustx.MODP, 0, 100, 0),
                new UExpressionDescriptor {
                    name = "breath low (curve)",
                    abbr = "brel",
                    type = UExpressionType.Curve,
                    min = -100,
                    max = 100,
                    defaultValue = 0,
                    isFlag = false,
                    skipOutputIfDefault = true,
                },
                new UExpressionDescriptor {
                    name = "breath high (curve)",
                    abbr = "breh",
                    type = UExpressionType.Curve,
                    min = -100,
                    max = 100,
                    defaultValue = 0,
                    isFlag = false,
                    skipOutputIfDefault = true,
                },
                new UExpressionDescriptor {
                    name = "brightness (curve)",
                    abbr = "bric",
                    type = UExpressionType.Curve,
                    min = -100,
                    max = 100,
                    defaultValue = 0,
                    isFlag = false,
                    skipOutputIfDefault = true,
                },
                new UExpressionDescriptor {
                    name = "growl (curve)",
                    abbr = "gwlc",
                    type = UExpressionType.Curve,
                    min = 0,
                    max = 100,
                    defaultValue = 0,
                    isFlag = false,
                    skipOutputIfDefault = true,
                },
            };
        }

        public override string ToString() => Renderers.HIFIUTAU;

        readonly struct PostProcessCurves {
            PostProcessCurves(
                float[]? breathiness,
                float[]? tension,
                float[]? voicing,
                float[]? brel,
                float[]? breh,
                float[]? bri,
                float[]? growl,
                bool needsHnsep,
                bool needsGrowl) {
                Breathiness = breathiness;
                Tension = tension;
                Voicing = voicing;
                Brel = brel;
                Breh = breh;
                Bri = bri;
                Growl = growl;
                NeedsHnsep = needsHnsep;
                NeedsGrowl = needsGrowl;
            }

            public readonly float[]? Breathiness;
            public readonly float[]? Tension;
            public readonly float[]? Voicing;
            public readonly float[]? Brel;
            public readonly float[]? Breh;
            public readonly float[]? Bri;
            public readonly float[]? Growl;
            public readonly bool NeedsHnsep;
            public readonly bool NeedsGrowl;

            public static PostProcessCurves FromPhrase(RenderPhrase phrase, HiFiUtauPhone[] phones) {
                var breathiness = MergePhoneValues(
                    phrase, phones, phrase.breathiness, phrase.breathinessCurveActive,
                    phone => phone.BreathinessValue, 0, -100, 100);
                var tension = MergePhoneValues(
                    phrase, phones, phrase.tension, phrase.tensionCurveActive,
                    phone => phone.TensionValue, 0, -100, 100);
                var voicing = MergePhoneValues(
                    phrase, phones, phrase.voicing, phrase.voicingCurveActive,
                    phone => phone.VoicingValue, 100, 0, 100);
                var brel = GetCurve(phrase, "brel");
                var breh = GetCurve(phrase, "breh");
                var bri = GetCurve(phrase, "bric");
                var growl = GetCurve(phrase, "gwlc");
                bool needsHnsep =
                    AudioPostProcessor.HasNonDefaultCurve(breathiness, 0, 0.5f) ||
                    AudioPostProcessor.HasNonDefaultCurve(tension, 0, 0.5f) ||
                    AudioPostProcessor.HasNonDefaultCurve(voicing, 100, 0.5f) ||
                    AudioPostProcessor.HasNonDefaultCurve(brel, 0, 0.5f) ||
                    AudioPostProcessor.HasNonDefaultCurve(breh, 0, 0.5f) ||
                    AudioPostProcessor.HasNonDefaultCurve(bri, 0, 0.5f);
                bool needsGrowl = AudioPostProcessor.HasNonDefaultCurve(growl, 0, 0.5f);
                return new PostProcessCurves(breathiness, tension, voicing, brel, breh, bri, growl, needsHnsep, needsGrowl);
            }

            static float[]? MergePhoneValues(
                RenderPhrase phrase,
                HiFiUtauPhone[] phones,
                float[]? phraseCurve,
                bool[] curveActive,
                Func<HiFiUtauPhone, float> valueSelector,
                float defaultValue,
                float min,
                float max) {
                if (!phones.Any(phone => Math.Abs(valueSelector(phone) - defaultValue) > 0.5f)) {
                    return phraseCurve;
                }
                if (phrase.pitches == null || phrase.pitches.Length == 0) {
                    return phraseCurve;
                }
                var curve = new float[phrase.pitches.Length];
                if (phraseCurve == null || phraseCurve.Length == 0) {
                    Array.Fill(curve, defaultValue);
                } else {
                    int copyLength = Math.Min(curve.Length, phraseCurve.Length);
                    Array.Copy(phraseCurve, curve, copyLength);
                    if (copyLength < curve.Length) {
                        Array.Fill(curve, defaultValue, copyLength, curve.Length - copyLength);
                    }
                }
                foreach (var phone in phones) {
                    float phoneValue = valueSelector(phone);
                    if (Math.Abs(phoneValue - defaultValue) <= 0.5f) {
                        continue;
                    }
                    int startTick = phrase.timeAxis.MsPosToTickPos(phone.PositionMs + phone.Envelope[0].X)
                        - (phrase.position - phrase.leading);
                    int endTick = phrase.timeAxis.MsPosToTickPos(phone.PositionMs + phone.Envelope[4].X)
                        - (phrase.position - phrase.leading);
                    int start = Math.Clamp(startTick / DynamicInterval, 0, curve.Length - 1);
                    int end = Math.Clamp((int)Math.Ceiling(endTick / (double)DynamicInterval), start + 1, curve.Length);
                    for (int i = start; i < end; i++) {
                        if (i >= curveActive.Length || !curveActive[i]) {
                            curve[i] = Math.Clamp(phoneValue, min, max);
                        }
                    }
                }
                return curve;
            }
        }

        static void WriteCurveActivity(BinaryWriter writer, bool[] activity) {
            writer.Write(activity.Length);
            foreach (var value in activity) {
                writer.Write(value);
            }
        }
    }
}

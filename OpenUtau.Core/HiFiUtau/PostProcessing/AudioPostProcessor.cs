using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using OpenUtau.Core.Format;
using OpenUtau.Core.Render;

namespace OpenUtau.Core.HiFiUtau {
    static class AudioPostProcessor {
        const int SampleRate = AudioPostProcessingDsp.SampleRate;
        const int FftSize = 2048;
        const int Hop = 512;

        static readonly object separatorLock = new object();
        static readonly Dictionary<string, HnsepSeparator> separators = new Dictionary<string, HnsepSeparator>();

        public static void Apply(RenderPhrase phrase, RenderResult result) {
            if (result.samples == null || result.samples.Length == 0) {
                return;
            }

            ApplyHnsepCurves(phrase, result.samples);
        }

        public static void ApplyWithSeparated(RenderPhrase phrase, RenderResult result, float[] harmonic, float[] noise,
            float[]? brelCurve = null, float[]? brehCurve = null, float[]? briCurve = null) {
            if (result.samples == null || result.samples.Length == 0) {
                return;
            }
            ApplyHnsepCurvesWithSeparated(phrase, result.samples, harmonic, noise, brelCurve, brehCurve, briCurve);
        }

        static void ApplyHnsepCurves(RenderPhrase phrase, float[] samples) {
            bool needBreath = HasNonDefaultCurve(phrase.breathiness, 0, 0.5f);
            bool needTension = HasNonDefaultCurve(phrase.tension, 0, 0.5f);
            bool needVoicing = HasNonDefaultCurve(phrase.voicing, 100, 0.5f);
            if (!needBreath && !needTension && !needVoicing) {
                return;
            }

            var hnsep = GetSeparator();

            var (harmonic, noise) = hnsep.Separate(samples);
            int length = Math.Min(samples.Length, Math.Min(harmonic.Length, noise.Length));
            if (length <= 0) {
                return;
            }

            ApplyCurvesToComponents(phrase, samples, harmonic, noise, needBreath, needTension, needVoicing, length);
        }

        static void ApplyHnsepCurvesWithSeparated(RenderPhrase phrase, float[] samples, float[] harmonic, float[] noise,
            float[]? brelCurve = null, float[]? brehCurve = null, float[]? briCurve = null) {
            bool needBreath = HasNonDefaultCurve(phrase.breathiness, 0, 0.5f);
            bool needTension = HasNonDefaultCurve(phrase.tension, 0, 0.5f);
            bool needVoicing = HasNonDefaultCurve(phrase.voicing, 100, 0.5f);
            bool needBrel = HasNonDefaultCurve(brelCurve, 0, 0.5f);
            bool needBreh = HasNonDefaultCurve(brehCurve, 0, 0.5f);
            bool needBri = HasNonDefaultCurve(briCurve, 0, 0.5f);
            if (!needBreath && !needTension && !needVoicing && !needBrel && !needBreh && !needBri) {
                return;
            }

            int length = Math.Min(samples.Length, Math.Min(harmonic.Length, noise.Length));
            if (length <= 0) {
                return;
            }

            ApplyCurvesToComponents(phrase, samples, harmonic, noise, needBreath, needTension, needVoicing, length, brelCurve, brehCurve, briCurve);
        }

        static void ApplyCurvesToComponents(RenderPhrase phrase, float[] samples, float[] harmonic, float[] noise,
            bool needBreath, bool needTension, bool needVoicing, int length,
            float[]? brelCurve = null, float[]? brehCurve = null, float[]? briCurve = null) {
            if (needBreath) {
                ApplyBreath(noise, phrase.breathiness, length);
            }
            if (needVoicing) {
                ApplyVoicing(harmonic, phrase.voicing, length);
            }
            if (needTension) {
                harmonic = ApplyTension(harmonic, phrase.tension, phrase);
                length = Math.Min(length, harmonic.Length);
            }
            bool hasBrel = HasNonDefaultCurve(brelCurve, 0, 0.5f);
            bool hasBreh = HasNonDefaultCurve(brehCurve, 0, 0.5f);
            if (hasBrel || hasBreh) {
                noise = ApplyBrelBreh(noise, brelCurve, brehCurve, length);
            }
            if (HasNonDefaultCurve(briCurve, 0, 0.5f)) {
                harmonic = ApplyBri(harmonic, briCurve, phrase);
                length = Math.Min(length, harmonic.Length);
            }

            MixComponents(samples, harmonic, noise, length);
        }

        public static HnsepSeparator GetSeparator() {
            if (!HnsepSeparator.TryResolveModelPath(out var modelPath)) {
                throw Error(
                    "HN-SEP model is required for HiFiUTAU BREC/TENC/VOIC.",
                    new FileNotFoundException("HN-SEP model package hnsep_VR_44.1k_hop512_240512 is not installed."),
                    false);
            }
            lock (separatorLock) {
                if (separators.TryGetValue(modelPath, out var separator)) {
                    return separator;
                }
                try {
                    separator = HnsepSeparator.Load(modelPath);
                } catch (Exception e) {
                    throw Error(
                        $"Failed to load HN-SEP model: {modelPath}",
                        e,
                        false);
                }
                separators[modelPath] = separator;
                return separator;
            }
        }

        static MessageCustomizableException Error(string message, Exception e, bool showStackTrace) {
            return new MessageCustomizableException(message, message, e, showStackTrace);
        }

        public static bool HasNonDefaultCurve(float[]? curve, float defaultValue, float tolerance) {
            return curve != null && curve.Any(value => Math.Abs(value - defaultValue) > tolerance);
        }

        static void ApplyBreath(float[] noise, float[]? breathiness, int length) {
            for (int i = 0; i < length; i++) {
                int idx = SampleIndexToCurveIndex(i, length, breathiness);
                double ratio = AudioPostProcessingDsp.CurveValue(breathiness, idx, 0) / 100.0;
                double gain = ratio > 0 ? 1.0 + ratio * 3.0 : 1.0 + ratio;
                noise[i] *= (float)Math.Clamp(gain, 0.0, 10.0);
            }
        }

        static void ApplyVoicing(float[] harmonic, float[]? voicing, int length) {
            for (int i = 0; i < length; i++) {
                int idx = SampleIndexToCurveIndex(i, length, voicing);
                double gain = Math.Clamp(AudioPostProcessingDsp.CurveValue(voicing, idx, 100), 0, 500) / 100.0;
                harmonic[i] *= (float)gain;
                if (gain < 1e-8) {
                    harmonic[i] = 0;
                }
            }
        }

        static float[] ApplyTension(float[] samples, float[]? tension, RenderPhrase phrase) {
            // 频谱倾斜锚定在约第四谐波，随音高变化
            int originalLength = samples.Length;
            int frames = Math.Max(1, (originalLength + Hop - 1) / Hop + 1);
            int bins = FftSize / 2 + 1;
            var tensionFrames = AudioPostProcessingDsp.ResampleCurve(tension, frames, 0);
            var f0Frames = AudioPostProcessingDsp.PitchHzCurve(phrase, frames);
            var window = AudioPostProcessingDsp.HannWindow(FftSize);
            var output = new float[originalLength + FftSize];
            var norm = new float[output.Length];
            var fft = AudioPostProcessingDsp.CreateRealFftWorkspace(FftSize);
            var frame = new float[FftSize];
            var spectrum = new Complex[bins];

            for (int frameIndex = 0; frameIndex < frames; frameIndex++) {
                float tv = tensionFrames[frameIndex];
                double b = tv > 0 ? -tv / 150.0 : -tv / 50.0;
                int outStart = frameIndex * Hop;
                ReadFrame(samples, window, outStart, frame);
                if (Math.Abs(b) < 0.001) {
                    OverlapAdd(frame, window, output, norm, outStart);
                    continue;
                }

                double midpointHz = f0Frames[frameIndex] < 30
                    ? 1500
                    : Math.Clamp(f0Frames[frameIndex] * 4.0, 400.0, 6000.0);
                double x0 = bins / ((SampleRate / 2.0) / midpointHz);
                AudioPostProcessingDsp.DirectRealFft(frame, spectrum, fft);

                var originalMag = new double[bins];
                double originalSum = 0;
                for (int bin = 0; bin < bins; bin++) {
                    originalMag[bin] = spectrum[bin].Magnitude;
                    originalSum += originalMag[bin];
                }

                double newSum = 0;
                for (int bin = 0; bin < bins; bin++) {
                    double tilt = Math.Clamp((-b / Math.Max(1.0, x0)) * bin + b, -2.0, 2.0);
                    double gain = Math.Exp(tilt);
                    spectrum[bin] *= gain;
                    newSum += originalMag[bin] * gain;
                }
                if (originalSum > 1e-12 && newSum > 1e-12) {
                    double comp = originalSum / newSum;
                    for (int bin = 0; bin < bins; bin++) {
                        spectrum[bin] *= comp;
                    }
                }
                if (b < -0.001) {
                    double bGain = 1.0 + Math.Clamp(b / -15.0, 0.0, 0.33);
                    for (int bin = 0; bin < bins; bin++) {
                        spectrum[bin] *= bGain;
                    }
                }

                AudioPostProcessingDsp.InverseRealFft(spectrum, frame, fft);
                OverlapAdd(frame, window, output, norm, outStart);
            }

            var result = new float[originalLength];
            for (int i = 0; i < result.Length; i++) {
                int src = i + FftSize / 2;
                result[i] = norm[src] > 1e-8 ? output[src] / norm[src] : 0;
            }
            SoftLimitIfClipped(result, result.Length);
            return result;
        }

        static void MixComponents(float[] samples, float[] harmonic, float[] noise, int length) {
            for (int i = 0; i < length; i++) {
                samples[i] = harmonic[i] + noise[i];
            }
            SoftLimitIfClipped(samples, length);
        }

        static void SoftLimitIfClipped(float[] samples, int length) {
            double peak = 0;
            for (int i = 0; i < length; i++) {
                peak = Math.Max(peak, Math.Abs(samples[i]));
            }
            if (peak <= 1.0) {
                return;
            }

            for (int i = 0; i < length; i++) {
                samples[i] = (float)(Math.Tanh(samples[i] * 0.9) / 0.9);
            }

            double newPeak = 0;
            for (int i = 0; i < length; i++) {
                newPeak = Math.Max(newPeak, Math.Abs(samples[i]));
            }
            if (newPeak <= 1e-8) {
                return;
            }
            double scale = Math.Min(peak, 1.0) / newPeak;
            for (int i = 0; i < length; i++) {
                samples[i] *= (float)scale;
            }
        }

        static int SampleIndexToCurveIndex(int sampleIndex, int sampleLength, float[]? curve) {
            if (curve == null || curve.Length <= 1 || sampleLength <= 1) {
                return 0;
            }
            return Math.Clamp((int)Math.Round(sampleIndex * (curve.Length - 1.0) / (sampleLength - 1)), 0, curve.Length - 1);
        }

        static void ReadFrame(float[] samples, float[] window, int outStart, float[] frame) {
            int start = outStart - FftSize / 2;
            for (int i = 0; i < FftSize; i++) {
                int src = AudioPostProcessingDsp.ReflectIndex(start + i, samples.Length);
                frame[i] = samples.Length == 0 ? 0 : samples[src] * window[i];
            }
        }

        static void OverlapAdd(float[] frame, float[] window, float[] output, float[] norm, int outStart) {
            for (int i = 0; i < FftSize; i++) {
                int dst = outStart + i;
                if (dst >= output.Length) {
                    break;
                }
                double w = window[i];
                output[dst] += (float)(frame[i] * w);
                norm[dst] += (float)(w * w);
            }
        }

        /// <summary>
        /// Apply BREL (低频气声) and BREH (高频气声) to noise component using STFT crossover at 2000Hz.
        /// BREL: gain on low band (&lt; 2000Hz). Positive: 1→×4, Negative: 1→0.
        /// BREH: gain on high band (&gt; 2000Hz). Same gain formula.
        /// Matching Python apply_breath_band_gain.
        /// </summary>
        static float[] ApplyBrelBreh(float[] noise, float[]? brelCurve, float[]? brehCurve, int length) {
            int originalLength = length;
            int frames = Math.Max(1, (originalLength + Hop - 1) / Hop + 1);
            int bins = FftSize / 2 + 1;
            double crossoverHz = 2000.0;
            int crossoverBin = (int)Math.Round(crossoverHz / (SampleRate / 2.0) * (bins - 1));
            crossoverBin = Math.Clamp(crossoverBin, 1, bins - 2);

            var window = AudioPostProcessingDsp.HannWindow(FftSize);
            var fft = AudioPostProcessingDsp.CreateRealFftWorkspace(FftSize);
            var frame = new float[FftSize];
            var spectrum = new Complex[bins];
            var output = new float[originalLength + FftSize];
            var norm = new float[output.Length];

            for (int frameIndex = 0; frameIndex < frames; frameIndex++) {
                int outStart = frameIndex * Hop;
                ReadFrame(noise, window, outStart, frame);
                AudioPostProcessingDsp.DirectRealFft(frame, spectrum, fft);

                // Per-frame gain values from curves
                int idx = SampleIndexToCurveIndex(outStart, originalLength, brelCurve);
                double brelRatio = AudioPostProcessingDsp.CurveValue(brelCurve, idx, 0) / 100.0;
                double brelGain = brelRatio > 0 ? 1.0 + brelRatio * 3.0 : 1.0 + brelRatio;
                brelGain = Math.Clamp(brelGain, 0.0, 10.0);

                idx = SampleIndexToCurveIndex(outStart, originalLength, brehCurve);
                double brehRatio = AudioPostProcessingDsp.CurveValue(brehCurve, idx, 0) / 100.0;
                double brehGain = brehRatio > 0 ? 1.0 + brehRatio * 3.0 : 1.0 + brehRatio;
                brehGain = Math.Clamp(brehGain, 0.0, 10.0);

                // Apply frequency-crossover gain
                for (int bin = 0; bin < bins; bin++) {
                    double blend = bin < crossoverBin
                        ? 1.0
                        : bin > crossoverBin + 1
                            ? 0.0
                            : 1.0 - (bin - crossoverBin) * 0.5;
                    double totalGain = brelGain * blend + brehGain * (1.0 - blend);
                    spectrum[bin] *= (float)totalGain;
                }

                AudioPostProcessingDsp.InverseRealFft(spectrum, frame, fft);
                OverlapAdd(frame, window, output, norm, outStart);
            }

            var result = new float[originalLength];
            for (int i = 0; i < result.Length; i++) {
                int src = i + FftSize / 2;
                result[i] = norm[src] > 1e-8 ? output[src] / norm[src] : 0;
            }
            return result;
        }

        /// <summary>
        /// Apply BRI (brightness/warmth, formerly warm) to harmonic component.
        /// Uses STFT EQ:
        ///   Positive BRI: warm mode — boost ~300Hz low band, gentle saturation-like effect
        ///   Negative BRI: bright mode — boost ~5kHz high band
        /// Matching Python apply_warmth_eq with simplified approach (EQ only).
        /// Range -100~100, 0 = bypass.
        /// </summary>
        static float[] ApplyBri(float[] harmonic, float[]? briCurve, RenderPhrase phrase) {
            int originalLength = harmonic.Length;
            int frames = Math.Max(1, (originalLength + Hop - 1) / Hop + 1);
            int bins = FftSize / 2 + 1;
            var briFrames = AudioPostProcessingDsp.ResampleCurve(briCurve, frames, 0);
            var window = AudioPostProcessingDsp.HannWindow(FftSize);
            var fft = AudioPostProcessingDsp.CreateRealFftWorkspace(FftSize);
            var frame = new float[FftSize];
            var spectrum = new Complex[bins];
            var output = new float[originalLength + FftSize];
            var norm = new float[output.Length];

            // Precompute frequency grid for bell curve
            var freqBin = new double[bins];
            for (int b = 0; b < bins; b++) {
                freqBin[b] = b * (SampleRate / 2.0) / (bins - 1);
            }

            for (int frameIndex = 0; frameIndex < frames; frameIndex++) {
                int outStart = frameIndex * Hop;
                float briValue = briFrames[frameIndex];
                if (Math.Abs(briValue) < 0.5f) {
                    ReadFrame(harmonic, window, outStart, frame);
                    OverlapAdd(frame, window, output, norm, outStart);
                    continue;
                }

                ReadFrame(harmonic, window, outStart, frame);
                AudioPostProcessingDsp.DirectRealFft(frame, spectrum, fft);

                double bri = -briValue / 100.0; // Positive = warm, negative = bright
                double centerHz = bri > 0 ? 300.0 : 5000.0;
                double sigmaOct = bri > 0 ? 2.4 : 3.0;
                double gainDb = bri > 0 ? bri * 6.0 : -bri * 8.0;

                double[] originalMag = new double[bins];
                double originalSum = 0;
                for (int b = 0; b < bins; b++) {
                    originalMag[b] = spectrum[b].Magnitude;
                    originalSum += originalMag[b];
                }

                for (int b = 0; b < bins; b++) {
                    double logF = freqBin[b] > 1.0 ? Math.Log2(freqBin[b]) : Math.Log2(1.0);
                    double logCenter = Math.Log2(centerHz);
                    double bell = Math.Exp(-0.5 * Math.Pow((logF - logCenter) / sigmaOct, 2));
                    double curve = 2.0 * bell - 1.0;
                    if (curve < 0) {
                        curve = -Math.Pow(-curve, 0.7);
                    }
                    double gainLinear = Math.Pow(10.0, (curve * gainDb) / 20.0);
                    spectrum[b] *= (float)gainLinear;
                }

                // Energy normalization
                double newSum = 0;
                for (int b = 0; b < bins; b++) {
                    newSum += spectrum[b].Magnitude;
                }
                if (originalSum > 1e-12 && newSum > 1e-12) {
                    double comp = originalSum / newSum;
                    for (int b = 0; b < bins; b++) {
                        spectrum[b] *= (float)comp;
                    }
                }

                AudioPostProcessingDsp.InverseRealFft(spectrum, frame, fft);
                OverlapAdd(frame, window, output, norm, outStart);
            }

            var result = new float[originalLength];
            for (int i = 0; i < result.Length; i++) {
                int src = i + FftSize / 2;
                result[i] = norm[src] > 1e-8 ? output[src] / norm[src] : 0;
            }
            SoftLimitIfClipped(result, result.Length);
            return result;
        }
    }
}

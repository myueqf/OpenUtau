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

        public static void Apply(RenderPhrase phrase, RenderResult result, string? modelLocation = null) {
            if (result.samples == null || result.samples.Length == 0) {
                return;
            }

            ApplyHnsepCurves(phrase, result.samples, modelLocation);
        }

        static void ApplyHnsepCurves(RenderPhrase phrase, float[] samples, string? modelLocation) {
            bool needBreath = HasNonDefaultCurve(phrase.breathiness, 0, 0.5f);
            bool needTension = HasNonDefaultCurve(phrase.tension, 0, 0.5f);
            bool needVoicing = HasNonDefaultCurve(phrase.voicing, 100, 0.5f);
            if (!needBreath && !needTension && !needVoicing) {
                return;
            }

            var hnsep = GetSeparator(modelLocation);

            var (harmonic, noise) = hnsep.Separate(samples);
            int length = Math.Min(samples.Length, Math.Min(harmonic.Length, noise.Length));
            if (length <= 0) {
                return;
            }

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

            MixComponents(samples, harmonic, noise, length);
        }

        static HnsepSeparator GetSeparator(string? modelLocation) {
            if (!HnsepSeparator.TryResolveModelPath(modelLocation, out var modelPath)) {
                throw Error(
                    "HN-SEP model is required for HiFiUTAU BREC/TENC/VOIC.",
                    new FileNotFoundException("HN-SEP model package hnsep-vr-44.1k-hop512 is not installed."),
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

        static bool HasNonDefaultCurve(float[]? curve, float defaultValue, float tolerance) {
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
    }
}

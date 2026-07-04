using System;
using System.Numerics;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using OpenUtau.Core.Format;

namespace OpenUtau.Core.HiFiUtau {
    static class HiFiUtauMath {
        public static int FramesForMs(double ms, double msPerFrame) {
            if (ms <= 0 || msPerFrame <= 0) {
                return 0;
            }
            return (int)Math.Round(ms / msPerFrame, MidpointRounding.AwayFromZero);
        }

        public static int ClampSample(double ms, int sampleRate, int length) {
            return Math.Clamp((int)Math.Round(ms * sampleRate / 1000.0), 0, length);
        }

        public static double StretchFactor(double velocity) {
            return Math.Pow(2.0, (100.0 - velocity) / 100.0);
        }

        public static float[] ReadMonoSamples(string path, int sampleRate) {
            if (string.IsNullOrEmpty(path)) {
                return Array.Empty<float>();
            }
            using var reader = Wave.OpenFile(path);
            if (reader == null) {
                return Array.Empty<float>();
            }
            ISampleProvider provider = reader.ToSampleProvider().ToMono(1, 0);
            if (reader.WaveFormat.SampleRate != sampleRate) {
                provider = new WdlResamplingSampleProvider(provider, sampleRate);
            }
            return ReadAll(provider);
        }

        public static float[] Resample(float[] samples, int sourceRate, int targetRate) {
            if (sourceRate == targetRate || samples.Length == 0) {
                return samples;
            }
            int targetLength = Math.Max(1, (int)Math.Round(samples.Length * targetRate / (double)sourceRate));
            var result = new float[targetLength];
            double ratio = sourceRate / (double)targetRate;
            for (int i = 0; i < targetLength; i++) {
                double src = Math.Clamp(i * ratio, 0, samples.Length - 1);
                int i0 = (int)Math.Floor(src);
                int i1 = Math.Min(samples.Length - 1, i0 + 1);
                float frac = (float)(src - i0);
                result[i] = samples[i0] + (samples[i1] - samples[i0]) * frac;
            }
            return result;
        }

        public static int ReflectIndex(int i, int length) {
            if (length <= 1) {
                return 0;
            }
            while (i < 0 || i >= length) {
                if (i < 0) {
                    i = -i;
                } else {
                    i = 2 * length - i - 2;
                }
            }
            return i;
        }

        public static float[,] SliceMel(float[,] mel, int start, int end) {
            int bins = mel.GetLength(0);
            int frames = Math.Max(0, end - start);
            var result = new float[bins, frames];
            for (int b = 0; b < bins; b++) {
                for (int t = 0; t < frames; t++) {
                    result[b, t] = mel[b, start + t];
                }
            }
            return result;
        }

        public static float[,] CreateBlankMel(int bins, int frames) {
            var result = new float[bins, frames];
            float blank = (float)Math.Log(1e-5);
            for (int b = 0; b < bins; b++) {
                for (int t = 0; t < frames; t++) {
                    result[b, t] = blank;
                }
            }
            return result;
        }

        public static float[,] PadBlankLeft(float[,] mel, int padFrames) {
            if (padFrames <= 0) {
                return mel;
            }
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var result = CreateBlankMel(bins, padFrames + frames);
            for (int b = 0; b < bins; b++) {
                for (int t = 0; t < frames; t++) {
                    result[b, padFrames + t] = mel[b, t];
                }
            }
            return result;
        }

        public static void AddLogGain(float[,] mel, double logGain) {
            for (int b = 0; b < mel.GetLength(0); b++) {
                for (int t = 0; t < mel.GetLength(1); t++) {
                    mel[b, t] += (float)logGain;
                }
            }
        }

        public static double MelRms(float[,] mel, int start = 0, int count = -1) {
            int bins = mel.GetLength(0);
            int frames = count < 0 ? mel.GetLength(1) - start : count;
            double sum = 0;
            int n = 0;
            for (int b = 0; b < bins; b++) {
                for (int t = start; t < start + frames; t++) {
                    double v = Math.Exp(mel[b, t]);
                    sum += v * v;
                    n++;
                }
            }
            return n == 0 ? 0 : Math.Sqrt(sum / n);
        }

        public static void WarpMelFrequency(float[,] mel, double factor) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var original = (float[,])mel.Clone();
            for (int t = 0; t < frames; t++) {
                for (int b = 0; b < bins; b++) {
                    double src = Math.Clamp(b / factor, 0, bins - 1);
                    int b0 = (int)Math.Floor(src);
                    int b1 = Math.Min(bins - 1, b0 + 1);
                    float frac = (float)(src - b0);
                    mel[b, t] = original[b0, t] + (original[b1, t] - original[b0, t]) * frac;
                }
            }
        }

        public static void WarpMelFrequency(float[,] mel, float[] semitoneCurve) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            if (semitoneCurve.Length == 0 || frames == 0) {
                return;
            }
            var original = (float[,])mel.Clone();
            for (int t = 0; t < frames; t++) {
                double semitones = ResampleCurve(semitoneCurve, t, frames);
                double factor = Math.Pow(2.0, semitones / 12.0);
                if (Math.Abs(factor - 1.0) < 0.001) {
                    continue;
                }
                for (int b = 0; b < bins; b++) {
                    double src = Math.Clamp(b / factor, 0, bins - 1);
                    int b0 = (int)Math.Floor(src);
                    int b1 = Math.Min(bins - 1, b0 + 1);
                    float frac = (float)(src - b0);
                    mel[b, t] = original[b0, t] + (original[b1, t] - original[b0, t]) * frac;
                }
            }
        }

        static double ResampleCurve(float[] curve, int frame, int targetFrames) {
            if (curve.Length == 1 || targetFrames == 1) {
                return curve[0];
            }
            double src = frame * (curve.Length - 1.0) / (targetFrames - 1);
            int i0 = (int)Math.Floor(src);
            int i1 = Math.Min(curve.Length - 1, i0 + 1);
            double frac = src - i0;
            return curve[i0] + (curve[i1] - curve[i0]) * frac;
        }

        public static void ApplyEnvelopeToMel(float[,] mel, Vector2[] envelope) {
            int totalFrames = mel.GetLength(1);
            if (totalFrames <= 0 || envelope == null || envelope.Length < 5) {
                return;
            }

            double envRange = envelope[4].X - envelope[0].X;
            if (envRange <= 0) {
                return;
            }

            // Map envelope points to frame indices (proportional mapping)
            double f1 = (envelope[1].X - envelope[0].X) / envRange * (totalFrames - 1);
            double f2 = (envelope[2].X - envelope[0].X) / envRange * (totalFrames - 1);
            double f3 = (envelope[3].X - envelope[0].X) / envRange * (totalFrames - 1);

            // Envelope rules:
            // p0→p1: crossfade region, use p1.Y constant
            // p1→p2: linear transition p1.Y → p2.Y
            // p2→p3: linear transition p2.Y → p3.Y
            // p3→p4: crossfade region, use p3.Y constant
            float y1 = envelope[1].Y / 100f;
            float y2 = envelope[2].Y / 100f;
            float y3 = envelope[3].Y / 100f;

            var gains = new float[totalFrames];
            for (int t = 0; t < totalFrames; t++) {
                float gain;
                if (t <= f1) {
                    gain = y1;
                } else if (t <= f2) {
                    double frac = f2 > f1 ? (t - f1) / (f2 - f1) : 1.0;
                    gain = y1 + (float)((y2 - y1) * frac);
                } else if (t <= f3) {
                    double frac = f3 > f2 ? (t - f2) / (f3 - f2) : 1.0;
                    gain = y2 + (float)((y3 - y2) * frac);
                } else {
                    gain = y3;
                }
                gains[t] = gain;
            }

            // Apply gains to mel (log domain)
            int bins = mel.GetLength(0);
            for (int t = 0; t < totalFrames; t++) {
                float logGain = (float)Math.Log(Math.Max(gains[t], 1e-10));
                for (int b = 0; b < bins; b++) {
                    mel[b, t] += logGain;
                }
            }
        }

        public static void ApplyPhraseEdgeEnvelope(HiFiUtauPhone[] phones, float[] samples, int sampleRate) {
            if (samples.Length == 0 || phones.Length == 0) {
                return;
            }
            // Fade-in: linear 0 → 1.0 across p0.X to p1.X
            var first = phones[0].Envelope;
            int fadeIn = Math.Max(0, (int)Math.Round((first[1].X - first[0].X) * sampleRate / 1000.0));
            fadeIn = Math.Min(fadeIn, samples.Length);
            if (fadeIn > 1) {
                for (int i = 0; i < fadeIn; i++) {
                    samples[i] *= (float)i / (fadeIn - 1);
                }
            }
            // Fade-out: linear 1.0 → 0 across p3.X to p4.X
            var last = phones[^1].Envelope;
            int fadeOut = Math.Max(0, (int)Math.Round((last[4].X - last[3].X) * sampleRate / 1000.0));
            fadeOut = Math.Min(fadeOut, samples.Length);
            if (fadeOut > 1) {
                for (int i = 0; i < fadeOut; i++) {
                    samples[samples.Length - fadeOut + i] *= (float)(fadeOut - 1 - i) / (fadeOut - 1);
                }
            }
        }

        public static float[,] ReflectPadVowel(float[,] mel, int consonantFrames, int padFrames) {
            if (padFrames <= 0) {
                return mel;
            }
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            int vowelFrames = frames - consonantFrames;
            if (vowelFrames <= 1) {
                return mel;
            }
            var result = new float[bins, frames + padFrames];
            Array.Copy(mel, result, mel.Length);
            for (int t = 0; t < padFrames; t++) {
                int period = 2 * vowelFrames - 2;
                int phase = t % period;
                int src = consonantFrames + (phase < vowelFrames ? phase : period - phase);
                for (int b = 0; b < bins; b++) {
                    result[b, frames + t] = mel[b, src];
                }
            }
            return result;
        }

        public static float[,] ResamplePhoneMel(float[,] mel, int totalFrames, int conFramesOrig, int targetConFrames, int vowFramesOrig, double stretch) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var result = new float[bins, totalFrames];
            int targetVowFrames = Math.Max(1, totalFrames - targetConFrames);
            for (int t = 0; t < totalFrames; t++) {
                double src = t < targetConFrames
                    ? t / stretch
                    : conFramesOrig + (t - targetConFrames) * (vowFramesOrig / (double)targetVowFrames);
                src = Math.Clamp(src, 0, frames - 1);
                int i0 = (int)Math.Floor(src);
                int i1 = Math.Min(frames - 1, i0 + 1);
                float frac = (float)(src - i0);
                for (int b = 0; b < bins; b++) {
                    result[b, t] = mel[b, i0] + (mel[b, i1] - mel[b, i0]) * frac;
                }
            }
            return result;
        }

        static float[] ReadAll(ISampleProvider provider) {
            var samples = new System.Collections.Generic.List<float>();
            var buffer = new float[Math.Max(1024, provider.WaveFormat.SampleRate)];
            int n;
            while ((n = provider.Read(buffer, 0, buffer.Length)) > 0) {
                for (int i = 0; i < n; i++) {
                    samples.Add(buffer[i]);
                }
            }
            return samples.ToArray();
        }
    }
}

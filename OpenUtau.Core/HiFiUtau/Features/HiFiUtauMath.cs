using System;
using System.Linq;
using System.Numerics;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using OpenUtau.Classic.Hifisampler;
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

        public static float[] HannWindow(int size) {
            return Enumerable.Range(0, size)
                .Select(i => (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (size - 1))))
                .ToArray();
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
            // Keep blank frames at the same mel floor as Hifisampler.
            float blank = (float)Math.Log(1e-9);
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

        public static void WarpMelFrequency(float[,] mel, float[] semitoneCurve, HiFiUtauConfig config) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            if (semitoneCurve.Length == 0 || frames == 0 || bins == 0) {
                return;
            }
            // Mel bins are not linear in Hz. Shift their physical center frequencies,
            // using the same Slaney scale and frequency limits as the extractor.
            var frequencies = HifiMelBasis.MelFrequencies(bins + 2, config.MelFMin, config.MelFMax);
            double minMel = HifiMelBasis.HzToMel(config.MelFMin);
            double melStep = (HifiMelBasis.HzToMel(config.MelFMax) - minMel) / (bins + 1);
            const double floor = 1e-9;
            var original = (float[,])mel.Clone();
            for (int t = 0; t < frames; t++) {
                double semitones = ResampleCurve(semitoneCurve, t, frames);
                double factor = Math.Pow(2.0, semitones / 12.0);
                if (Math.Abs(factor - 1.0) < 0.001) {
                    continue;
                }
                for (int b = 0; b < bins; b++) {
                    double sourceHz = frequencies[b + 1] / factor;
                    if (sourceHz < config.MelFMin || sourceHz > config.MelFMax) {
                        // Out-of-band energy is absent, not a repeated edge band.
                        mel[b, t] = (float)Math.Log(floor);
                        continue;
                    }
                    double src = Math.Clamp((HifiMelBasis.HzToMel(sourceHz) - minMel) / melStep - 1, 0, bins - 1);
                    int b0 = (int)Math.Floor(src);
                    int b1 = Math.Min(bins - 1, b0 + 1);
                    double frac = src - b0;
                    // Mel stores log magnitudes; interpolate magnitudes before compression.
                    double magnitude = Math.Exp(original[b0, t]) * (1 - frac) + Math.Exp(original[b1, t]) * frac;
                    mel[b, t] = (float)Math.Log(Math.Max(magnitude, floor));
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

        /// <summary>
        /// Smooths the frame energy of a looped vowel to the linear trend between
        /// its first and last frames. This matches Fragment's strt=1 processing.
        /// </summary>
        public static void NormalizeLoopVowelEnergy(float[,] mel, int consonantFrames) {
            int bins = mel.GetLength(0);
            int totalFrames = mel.GetLength(1);
            consonantFrames = Math.Clamp(consonantFrames, 0, totalFrames);
            int vowelFrames = totalFrames - consonantFrames;
            if (bins == 0 || vowelFrames <= 1) {
                return;
            }

            var frameEnergy = new double[vowelFrames];
            for (int t = 0; t < vowelFrames; t++) {
                double sum = 0;
                for (int b = 0; b < bins; b++) {
                    sum += Math.Exp(mel[b, consonantFrames + t]);
                }
                frameEnergy[t] = Math.Max(sum / bins, 1e-12);
            }

            for (int t = 0; t < vowelFrames; t++) {
                double fraction = t / (double)(vowelFrames - 1);
                double targetEnergy = frameEnergy[0] +
                    (frameEnergy[vowelFrames - 1] - frameEnergy[0]) * fraction;
                float correction = (float)(Math.Log(targetEnergy) - Math.Log(frameEnergy[t]));
                for (int b = 0; b < bins; b++) {
                    mel[b, consonantFrames + t] += correction;
                }
            }
        }

        /// <summary>
        /// Pads silence on the left and fades its last frames into the first mel
        /// frame, matching Fragment's protection against hard decoder transitions.
        /// </summary>
        public static float[,] PadBlankLeftFadeIn(float[,] mel, int padFrames) {
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

            int fadeFrames = Math.Min(12, padFrames);
            if (frames > 0) {
                for (int t = 0; t < fadeFrames; t++) {
                    int target = padFrames - fadeFrames + t;
                    double alpha = (t + 1.0) / (fadeFrames + 1.0);
                    for (int b = 0; b < bins; b++) {
                        result[b, target] = (float)Math.Log(Math.Exp(mel[b, 0]) * alpha + 1e-10);
                    }
                }
            }
            return result;
        }

        internal static float PhoneFadeGain(HiFiUtauPhone phone, double positionMs) {
            var envelope = phone.Envelope;
            double x = positionMs - phone.PositionMs;
            if (x <= envelope[0].X || x >= envelope[4].X) {
                return 0;
            }
            if (x < envelope[1].X) {
                return (float)((x - envelope[0].X) / (envelope[1].X - envelope[0].X));
            }
            if (x > envelope[3].X) {
                return (float)((envelope[4].X - x) / (envelope[4].X - envelope[3].X));
            }
            return 1;
        }

        public static float[] BuildPhoneFadeEnvelope(HiFiUtauPhone[] phones, int sampleCount, int sampleRate,
            double phraseStartMs) {
            var gains = new float[sampleCount];
            foreach (var phone in phones) {
                int start = ClampSample(phone.PositionMs + phone.Envelope[0].X - phraseStartMs,
                    sampleRate, sampleCount);
                int end = ClampSample(phone.PositionMs + phone.Envelope[4].X - phraseStartMs,
                    sampleRate, sampleCount);
                for (int i = start; i < end; i++) {
                    // Classic adds independently enveloped phones. Their gains need not
                    // sum to one when the attack/release handles have been moved.
                    // ATK/DEC levels are already in the mel; only apply the time ramps here.
                    gains[i] += PhoneFadeGain(phone, phraseStartMs + i * 1000.0 / sampleRate);
                }
            }
            return gains;
        }

        public static float[,] ReflectPadVowel(float[,] mel, int consonantFrames, int padFrames) {
            if (padFrames <= 0) {
                return mel;
            }
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            consonantFrames = Math.Clamp(consonantFrames, 0, frames);
            int vowelFrames = frames - consonantFrames;
            if (vowelFrames <= 1) {
                return mel;
            }
            var result = new float[bins, frames + padFrames];
            // Rectangular arrays use the full second dimension as their row stride,
            // so a flat Array.Copy would put source rows into the wrong destinations.
            for (int b = 0; b < bins; b++) {
                for (int t = 0; t < frames; t++) {
                    result[b, t] = mel[b, t];
                }
            }
            // numpy.pad(vowel, (0, padFrames), mode='reflect') continues from
            // the second-to-last vowel frame, excluding each reflected edge.
            int period = 2 * vowelFrames - 2;
            for (int t = 0; t < padFrames; t++) {
                int p = t % period;
                int src;
                if (p < vowelFrames - 1) {
                    // Descending: v-2, v-3, ..., 0
                    src = consonantFrames + (vowelFrames - 2 - p);
                } else {
                    // Ascending: 1, 2, ..., v-1
                    src = consonantFrames + (p - (vowelFrames - 2));
                }
                for (int b = 0; b < bins; b++) {
                    result[b, frames + t] = mel[b, src];
                }
            }
            return result;
        }

        /// <summary>
        /// Loop-mode mel processing matching HiFiUTAU-engine Fragment. Long vowels
        /// are reflect-padded, linearly mapped to the note duration, then normalized
        /// to remove periodic energy changes introduced by the reflected material.
        /// </summary>
        public static float[,] ResamplePhoneMelLoop(float[,] mel, int totalFrames, int conFramesOrig, int targetConFrames, int vowFramesOrig, double stretch) {
            totalFrames = Math.Max(1, totalFrames);
            targetConFrames = Math.Clamp(targetConFrames, 0, totalFrames);
            int bins = mel.GetLength(0);
            if (bins == 0 || mel.GetLength(1) == 0) {
                return new float[bins, totalFrames];
            }
            int targetVowFrames = totalFrames - targetConFrames;
            if (vowFramesOrig > 1 && targetVowFrames > vowFramesOrig * 1.5) {
                int padExtra = Math.Min(4, vowFramesOrig / 2);
                int padFrames = targetVowFrames - vowFramesOrig + padExtra;
                mel = ReflectPadVowel(mel, conFramesOrig, padFrames);
                vowFramesOrig = mel.GetLength(1) - conFramesOrig;
            }

            var result = ResamplePhoneMel(
                mel, totalFrames, conFramesOrig, targetConFrames, vowFramesOrig, stretch);
            if (targetVowFrames > 1 && result.GetLength(1) >= targetConFrames + 2) {
                NormalizeLoopVowelEnergy(result, targetConFrames);
            }
            return result;
        }

        public static float[,] ResamplePhoneMel(float[,] mel, int totalFrames, int conFramesOrig, int targetConFrames, int vowFramesOrig, double stretch) {
            totalFrames = Math.Max(1, totalFrames);
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var result = new float[bins, totalFrames];
            targetConFrames = Math.Clamp(targetConFrames, 0, totalFrames);
            int targetVowFrames = totalFrames - targetConFrames;
            for (int t = 0; t < totalFrames; t++) {
                double src;
                if (t < targetConFrames) {
                    src = t / stretch;
                } else if (targetVowFrames > 0) {
                    src = conFramesOrig + (t - targetConFrames) * (vowFramesOrig / (double)targetVowFrames);
                } else {
                    // No vowel frames remain. Do not force a jump to the source
                    // tail; the fixed-region prefix above is the complete output.
                    src = t / stretch;
                }
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

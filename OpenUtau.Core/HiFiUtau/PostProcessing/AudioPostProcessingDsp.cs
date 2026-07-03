using System;
using System.Linq;
using System.Numerics;
using OpenUtau.Core.Render;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.HiFiUtau {
    static class AudioPostProcessingDsp {
        public const int SampleRate = 44100;

        public static float[] HannWindow(int size) {
            return Enumerable.Range(0, size)
                .Select(i => (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (size - 1))))
                .ToArray();
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

        public static float CurveValue(float[]? curve, int index, float defaultValue) {
            if (curve == null || curve.Length == 0) {
                return defaultValue;
            }
            return curve[Math.Clamp(index, 0, curve.Length - 1)];
        }

        public static float[] ResampleCurve(float[]? curve, int length, float defaultValue) {
            var result = new float[length];
            if (length == 0) {
                return result;
            }
            if (curve == null || curve.Length == 0) {
                Array.Fill(result, defaultValue);
                return result;
            }
            if (curve.Length == 1 || length == 1) {
                Array.Fill(result, curve[0]);
                return result;
            }
            for (int i = 0; i < length; i++) {
                double src = i * (curve.Length - 1.0) / (length - 1);
                int i0 = (int)Math.Floor(src);
                int i1 = Math.Min(curve.Length - 1, i0 + 1);
                float frac = (float)(src - i0);
                result[i] = curve[i0] + (curve[i1] - curve[i0]) * frac;
            }
            return result;
        }

        public static float[] PitchHzCurve(RenderPhrase phrase, int length) {
            var result = new float[length];
            if (length == 0) {
                return result;
            }
            if (phrase.pitches == null || phrase.pitches.Length == 0) {
                Array.Fill(result, 120f);
                return result;
            }
            for (int i = 0; i < length; i++) {
                int idx = length == 1
                    ? 0
                    : Math.Clamp((int)Math.Round(i * (phrase.pitches.Length - 1.0) / (length - 1)), 0, phrase.pitches.Length - 1);
                result[i] = (float)MusicMath.ToneToFreq(phrase.pitches[idx] * 0.01);
            }
            return result;
        }

        public static void Fft(Complex[] buffer, bool inverse) {
            int n = buffer.Length;
            for (int i = 1, j = 0; i < n; i++) {
                int bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1) {
                    j ^= bit;
                }
                j ^= bit;
                if (i < j) {
                    (buffer[i], buffer[j]) = (buffer[j], buffer[i]);
                }
            }
            for (int len = 2; len <= n; len <<= 1) {
                double angle = 2 * Math.PI / len * (inverse ? 1 : -1);
                var wLen = new Complex(Math.Cos(angle), Math.Sin(angle));
                for (int i = 0; i < n; i += len) {
                    var w = Complex.One;
                    for (int j = 0; j < len / 2; j++) {
                        var u = buffer[i + j];
                        var v = buffer[i + j + len / 2] * w;
                        buffer[i + j] = u + v;
                        buffer[i + j + len / 2] = u - v;
                        w *= wLen;
                    }
                }
            }
            if (inverse) {
                for (int i = 0; i < n; i++) {
                    buffer[i] /= n;
                }
            }
        }
    }
}

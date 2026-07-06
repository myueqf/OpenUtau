using System;
using System.Numerics;
using NWaves.Transforms;
using OpenUtau.Core.Render;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.HiFiUtau {
    static class AudioPostProcessingDsp {
        public const int SampleRate = 44100;

        public static float[] HannWindow(int size) {
            return HiFiUtauMath.HannWindow(size);
        }

        public static int ReflectIndex(int i, int length) {
            return HiFiUtauMath.ReflectIndex(i, length);
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

        public static RealFftWorkspace CreateRealFftWorkspace(int fftSize) {
            return new RealFftWorkspace(fftSize);
        }

        public static void DirectRealFft(float[] frame, Complex[] spectrum, RealFftWorkspace workspace) {
            workspace.Transform.Direct(frame, workspace.Re, workspace.Im);
            for (int bin = 0; bin < workspace.Bins; bin++) {
                spectrum[bin] = new Complex(workspace.Re[bin], workspace.Im[bin]);
            }
        }

        public static void InverseRealFft(Complex[] spectrum, float[] frame, RealFftWorkspace workspace) {
            for (int bin = 0; bin < workspace.Bins; bin++) {
                workspace.Re[bin] = (float)spectrum[bin].Real;
                workspace.Im[bin] = (float)spectrum[bin].Imaginary;
            }
            workspace.Transform.InverseNorm(workspace.Re, workspace.Im, frame);
        }

        public sealed class RealFftWorkspace {
            public RealFftWorkspace(int fftSize) {
                Transform = new RealFft(fftSize);
                Bins = fftSize / 2 + 1;
                Re = new float[Bins];
                Im = new float[Bins];
            }

            public readonly RealFft Transform;
            public readonly int Bins;
            public readonly float[] Re;
            public readonly float[] Im;
        }
    }
}

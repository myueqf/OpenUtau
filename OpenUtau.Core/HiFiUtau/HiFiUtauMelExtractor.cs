using System;
using System.Linq;
using NWaves.Transforms;

namespace OpenUtau.Core.HiFiUtau {
    class HiFiUtauMelExtractor {
        readonly HiFiUtauConfig config;
        readonly RealFft fft;
        readonly float[] window;
        readonly float[,] melBasis;

        public HiFiUtauMelExtractor(HiFiUtauConfig config) {
            this.config = config;
            fft = new RealFft(config.FftSize);
            window = Enumerable.Range(0, config.WinSize)
                .Select(i => (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (config.WinSize - 1))))
                .ToArray();
            melBasis = CreateMelBasis(config);
        }

        public float[,] Extract(float[] audio, int[] centers) {
            int bins = config.FftSize / 2 + 1;
            var result = new float[config.NumMels, centers.Length];
            var frame = new float[config.FftSize];
            var re = new float[bins];
            var im = new float[bins];
            for (int t = 0; t < centers.Length; t++) {
                Array.Clear(frame, 0, frame.Length);
                int start = centers[t] - config.FftSize / 2;
                for (int i = 0; i < config.WinSize; i++) {
                    int src = HiFiUtauMath.ReflectIndex(start + i, audio.Length);
                    frame[i] = audio.Length == 0 ? 0 : audio[src] * window[i];
                }
                fft.Direct(frame, re, im);
                for (int m = 0; m < config.NumMels; m++) {
                    double sum = 0;
                    for (int b = 0; b < bins; b++) {
                        double mag = Math.Sqrt(re[b] * re[b] + im[b] * im[b]);
                        sum += melBasis[m, b] * mag;
                    }
                    result[m, t] = (float)Math.Log(Math.Max(sum, 1e-5));
                }
            }
            return result;
        }

        static float[,] CreateMelBasis(HiFiUtauConfig config) {
            int fftBins = config.FftSize / 2 + 1;
            var basis = new float[config.NumMels, fftBins];
            double minMel = HzToSlaneyMel(config.MelFMin);
            double maxMel = HzToSlaneyMel(config.MelFMax);
            var melPoints = Enumerable.Range(0, config.NumMels + 2)
                .Select(i => minMel + (maxMel - minMel) * i / (config.NumMels + 1))
                .Select(SlaneyMelToHz)
                .ToArray();
            var bins = melPoints
                .Select(hz => (int)Math.Floor((config.FftSize + 1) * hz / config.SampleRate))
                .Select(bin => Math.Clamp(bin, 0, fftBins - 1))
                .ToArray();
            for (int m = 1; m <= config.NumMels; m++) {
                int left = bins[m - 1];
                int center = bins[m];
                int right = bins[m + 1];
                for (int k = left; k < center; k++) {
                    basis[m - 1, k] = (k - left) / (float)Math.Max(1, center - left);
                }
                for (int k = center; k < right; k++) {
                    basis[m - 1, k] = (right - k) / (float)Math.Max(1, right - center);
                }
                double enorm = 2.0 / Math.Max(1e-12, melPoints[m + 1] - melPoints[m - 1]);
                for (int k = left; k < right; k++) {
                    basis[m - 1, k] *= (float)enorm;
                }
            }
            return basis;
        }

        static double HzToSlaneyMel(double hz) {
            const double fSp = 200.0 / 3;
            double mel = hz / fSp;
            const double minLogHz = 1000.0;
            const double minLogMel = minLogHz / fSp;
            double logStep = Math.Log(6.4) / 27.0;
            if (hz >= minLogHz) {
                mel = minLogMel + Math.Log(hz / minLogHz) / logStep;
            }
            return mel;
        }

        static double SlaneyMelToHz(double mel) {
            const double fSp = 200.0 / 3;
            const double minLogHz = 1000.0;
            const double minLogMel = minLogHz / fSp;
            double logStep = Math.Log(6.4) / 27.0;
            if (mel >= minLogMel) {
                return minLogHz * Math.Exp(logStep * (mel - minLogMel));
            }
            return fSp * mel;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using NWaves.Transforms;

namespace OpenUtau.Core.HiFiUtau {
    class HiFiUtauMelExtractor {
        readonly HiFiUtauConfig config;
        readonly RealFft fft;
        readonly float[] window;
        readonly MelBand[] melBands;

        public HiFiUtauMelExtractor(HiFiUtauConfig config) {
            this.config = config;
            fft = new RealFft(config.FftSize);
            window = Enumerable.Range(0, config.WinSize)
                .Select(i => (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (config.WinSize - 1))))
                .ToArray();
            melBands = CreateMelBands(config);
        }

        public float[,] Extract(float[] audio, int[] centers) {
            int bins = config.FftSize / 2 + 1;
            var result = new float[config.NumMels, centers.Length];
            var frame = new float[config.FftSize];
            var re = new float[bins];
            var im = new float[bins];
            var mag = new float[bins];
            for (int t = 0; t < centers.Length; t++) {
                Array.Clear(frame, 0, frame.Length);
                int start = centers[t] - config.FftSize / 2;
                for (int i = 0; i < config.WinSize; i++) {
                    int src = HiFiUtauMath.ReflectIndex(start + i, audio.Length);
                    frame[i] = audio.Length == 0 ? 0 : audio[src] * window[i];
                }
                fft.Direct(frame, re, im);
                for (int b = 0; b < bins; b++) {
                    mag[b] = (float)Math.Sqrt(re[b] * re[b] + im[b] * im[b]);
                }
                for (int m = 0; m < config.NumMels; m++) {
                    var band = melBands[m];
                    var weights = band.Weights;
                    double sum = 0;
                    for (int i = 0; i < weights.Length; i++) {
                        sum += weights[i] * mag[band.Start + i];
                    }
                    result[m, t] = (float)Math.Log(Math.Max(sum, 1e-5));
                }
            }
            return result;
        }

        static MelBand[] CreateMelBands(HiFiUtauConfig config) {
            int fftBins = config.FftSize / 2 + 1;
            var bands = new MelBand[config.NumMels];
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
                double enorm = 2.0 / Math.Max(1e-12, melPoints[m + 1] - melPoints[m - 1]);
                var weights = new List<float>(Math.Max(0, right - left));
                for (int k = left; k < right; k++) {
                    float weight = 0;
                    if (k < center) {
                        weight = (k - left) / (float)Math.Max(1, center - left);
                    } else {
                        weight = (right - k) / (float)Math.Max(1, right - center);
                    }
                    weight *= (float)enorm;
                    weights.Add(weight);
                }
                while (weights.Count > 0 && weights[0] == 0) {
                    weights.RemoveAt(0);
                    left++;
                }
                while (weights.Count > 0 && weights[^1] == 0) {
                    weights.RemoveAt(weights.Count - 1);
                }
                bands[m - 1] = new MelBand(left, weights.ToArray());
            }
            return bands;
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

        readonly struct MelBand {
            public MelBand(int start, float[] weights) {
                Start = start;
                Weights = weights;
            }

            public readonly int Start;
            public readonly float[] Weights;
        }
    }
}

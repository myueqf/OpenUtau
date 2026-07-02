using System;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace OpenUtau.Core.HiFiUtau {
    class HiFiUtauConfig {
        public const int OutputSampleRate = 44100;

        public int SampleRate { get; private set; } = OutputSampleRate;
        public int ModelHop { get; private set; } = 512;
        public int FeatureHop { get; private set; } = 64;
        public int FftSize { get; private set; } = 2048;
        public int WinSize { get; private set; } = 2048;
        public int NumMels { get; private set; } = 128;
        public double MelFMin { get; private set; } = 40;
        public double MelFMax { get; private set; } = 16000;
        public int FeatUpsample { get; private set; } = 64;
        public int EncoderPadFrames { get; private set; } = 8;
        public int FrontPadFrames { get; private set; } = 6;
        public int TailPadFrames { get; private set; } = 4;

        public double MsPerModelFrame => ModelHop * 1000.0 / SampleRate;
        public double MsPerFeatureFrame => FeatureHop * 1000.0 / SampleRate;

        public static HiFiUtauConfig Load(string configPath) {
            var config = new HiFiUtauConfig();
            using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
            var root = doc.RootElement;
            config.SampleRate = GetInt(root, "sampling_rate", config.SampleRate);
            config.ModelHop = GetInt(root, "hop_size", config.ModelHop);
            config.FftSize = GetInt(root, "n_fft", config.FftSize);
            config.WinSize = GetInt(root, "win_size", config.WinSize);
            config.NumMels = GetInt(root, "num_mels", config.NumMels);
            config.MelFMin = GetDouble(root, "fmin", config.MelFMin);
            config.MelFMax = GetDouble(root, "fmax", config.MelFMax);
            config.FeatureHop = GetInt(root, "feature_hop_size", config.ModelHop / 8);
            config.FeatUpsample = GetInt(root, "feat_upsample", config.FeatUpsample);
            config.EncoderPadFrames = GetInt(root, "encoder_pad_frames", config.EncoderPadFrames);
            config.FrontPadFrames = GetInt(root, "front_pad_frames", config.FrontPadFrames);
            config.TailPadFrames = GetInt(root, "tail_pad_frames", config.TailPadFrames);
            if (root.TryGetProperty("upsample_rates", out var upsampleRates) &&
                upsampleRates.ValueKind == JsonValueKind.Array) {
                var product = upsampleRates.EnumerateArray()
                    .Select(value => value.TryGetInt32(out var rate) ? rate : 1)
                    .Aggregate(1, (a, b) => a * b);
                if (product > 0) {
                    config.ModelHop = product;
                }
            }
            return config;
        }

        static int GetInt(JsonElement root, string name, int defaultValue) {
            return root.TryGetProperty(name, out var value) && value.TryGetInt32(out var result)
                ? result
                : defaultValue;
        }

        static double GetDouble(JsonElement root, string name, double defaultValue) {
            return root.TryGetProperty(name, out var value) && value.TryGetDouble(out var result)
                ? result
                : defaultValue;
        }
    }
}

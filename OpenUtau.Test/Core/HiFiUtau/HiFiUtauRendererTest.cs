using System;
using System.IO;
using System.Linq;
using System.Numerics;
using NAudio.Wave;
using OpenUtau.Core.Ustx;
using Xunit;

namespace OpenUtau.Core.HiFiUtau {
    public class HiFiUtauRendererTest {
        [Theory]
        [InlineData(100, 0, 0)]
        [InlineData(50, 0, 0)]
        [InlineData(100, 25, 0)]
        [InlineData(50, 25, 0)]
        [InlineData(100, -20, 0)]
        [InlineData(50, 0, 1)]
        [InlineData(50, 25, 1)]
        [InlineData(100, -20, 1)]
        public void ExtractFeatureMel_KeepsConsonantOnClassicTimeline(
                double velocity, double leadingDelta, int stretchMode) {
            var config = new HiFiUtauConfig();
            var source = new float[config.SampleRate];
            const double offsetMs = 100;
            const double transientMs = 60;
            source[(int)Math.Round((offsetMs + transientMs) * config.SampleRate / 1000.0)] = 1;
            var path = Path.Combine(Path.GetTempPath(), $"hifi-timing-{Guid.NewGuid():N}.wav");
            try {
                using (var writer = new WaveFileWriter(path,
                        WaveFormat.CreateIeeeFloatWaveFormat(config.SampleRate, 1))) {
                    writer.WriteSamples(source, 0, source.Length);
                }
                double stretch = Math.Pow(2, (100 - velocity) / 100);
                // 40.2 ms rounds upward in float. The old negative-skip branch
                // inserted 12 source frames even without a user timing adjustment.
                double leadingMs = 40.2 * stretch + leadingDelta;
                var phone = new HiFiUtauPhone {
                    AudioPath = path,
                    OffsetMs = offsetMs,
                    ConsonantMs = 120,
                    CutoffMs = -400,
                    PreutterMs = 40.2,
                    LeadingMs = leadingMs,
                    Velocity = velocity,
                    StretchMode = stretchMode,
                    Envelope = [new((float)-leadingMs, 0), new(0, 100),
                        new(0, 100), new(465, 100), new(500, 0)],
                };

                var mel = HiFiUtauRenderer.ExtractFeatureMel(phone, config,
                    new HiFiUtauMelExtractor(config), new HiFiUtauLoudnessNormalizer());
                int peakFrame = Enumerable.Range(0, mel.GetLength(1))
                    .MaxBy(t => Enumerable.Range(0, mel.GetLength(0)).Sum(b => Math.Exp(mel[b, t])));
                // Classic places source audio at position - leading - skipOver.
                double skipOverMs = phone.PreutterMs * stretch - leadingMs;
                double actualMs = -leadingMs + peakFrame * config.MsPerFeatureFrame;
                double expectedMs = -leadingMs - skipOverMs + transientMs * stretch;
                Assert.InRange(Math.Abs(actualMs - expectedMs), 0, 2 * config.MsPerFeatureFrame);
            } finally {
                File.Delete(path);
            }
        }

        [Theory]
        [InlineData(0.0, 0.0)]
        [InlineData(0.5, 0.5)]
        [InlineData(1.0, 1.0)]
        [InlineData(2.0, 2.0)]
        public void ApplyPhoneVolumes_MatchesClassicLinearRatio(double volume, double expected) {
            var samples = new float[8];
            Array.Fill(samples, 1f);
            var phones = new[] {
                new HiFiUtauPhone {
                    ModelStartFrame = 2,
                    ModelEndFrame = 6,
                    Volume = volume,
                },
            };

            HiFiUtauRenderer.ApplyPhoneVolumes(samples, phones, 1.0);

            Assert.All(samples, sample => Assert.InRange(Math.Abs(sample - expected), 0, 1e-6));
        }

        [Fact]
        public void ApplyPhoneVolumes_CrossfadesOverlappingPhones() {
            var samples = new float[10];
            Array.Fill(samples, 1f);
            var phones = new[] {
                new HiFiUtauPhone {
                    ModelStartFrame = 0,
                    ModelEndFrame = 6,
                    Volume = 0.5,
                },
                new HiFiUtauPhone {
                    ModelStartFrame = 4,
                    ModelEndFrame = 10,
                    Volume = 1.0,
                },
            };

            HiFiUtauRenderer.ApplyPhoneVolumes(samples, phones, 1.0);

            Assert.Equal(new float[] {
                0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            }, samples);
        }

        [Fact]
        public void SuggestedExpressions_ContainsDirectOption() {
            var renderer = new HiFiUtauRenderer();
            var descriptor = renderer
                .GetSuggestedExpressions(null, null)
                .Single(expression => expression.abbr == OpenUtau.Core.Format.Ustx.DIR);

            Assert.True(renderer.SupportsExpression(descriptor));
            Assert.Equal("direct", descriptor.name);
            Assert.Equal(UExpressionType.Options, descriptor.type);
            Assert.Equal(new[] { "off", "on" }, descriptor.options);
            Assert.False(descriptor.isFlag);
        }

        [Fact]
        public void SliceDirectSamples_UsesOtoOffsetAndCutoff() {
            var source = Enumerable.Range(0, 100).Select(value => (float)value).ToArray();
            var phone = new HiFiUtauPhone {
                OffsetMs = 10,
                CutoffMs = 20,
            };

            var result = HiFiUtauRenderer.SliceDirectSamples(source, phone, 1000);

            Assert.Equal(70, result.Length);
            Assert.Equal(10f, result[0]);
            Assert.Equal(79f, result[^1]);
        }

        [Fact]
        public void ApplyDirectPhone_UsesClassicSkipOver() {
            var destination = new float[4];
            var source = Enumerable.Range(0, 14).Select(value => (float)value).ToArray();
            var phone = new HiFiUtauPhone {
                PositionMs = 20,
                LeadingMs = 20,
                PreutterMs = 30,
                Velocity = 100,
                Volume = 1,
                Envelope = new[] {
                    new Vector2(-20, 100),
                    new Vector2(-10, 100),
                    new Vector2(0, 100),
                    new Vector2(10, 100),
                    new Vector2(20, 100),
                },
            };

            HiFiUtauRenderer.ApplyDirectPhone(destination, source, phone, 0, 1000);

            Assert.Equal(new[] { 10f, 11f, 12f, 13f }, destination);
        }

        [Fact]
        public void ApplyDirectPhone_DoesNotEraseAudioAfterFadeOut() {
            var destination = Enumerable.Repeat(10f, 5).ToArray();
            var source = Enumerable.Repeat(2f, 5).ToArray();
            var phone = new HiFiUtauPhone {
                PositionMs = 0,
                LeadingMs = 0,
                PreutterMs = 0,
                Velocity = 100,
                Volume = 1,
                Envelope = new[] {
                    new Vector2(0, 100),
                    new Vector2(1, 100),
                    new Vector2(2, 100),
                    new Vector2(3, 50),
                    new Vector2(4, 0),
                },
            };

            HiFiUtauRenderer.ApplyDirectPhone(destination, source, phone, 0, 1000);

            Assert.Equal(2f, destination[0]);
            Assert.Equal(6f, destination[3]);
            Assert.Equal(10f, destination[4]);
        }
    }
}

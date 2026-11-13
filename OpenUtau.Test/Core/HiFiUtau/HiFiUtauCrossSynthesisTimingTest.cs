using System.Linq;
using OpenUtau.Core.Ustx;
using Xunit;

namespace OpenUtau.Core.HiFiUtau {
    public class HiFiUtauCrossSynthesisTimingTest {
        [Theory]
        [InlineData(100, 20, 10)]
        [InlineData(20, 100, 40)]
        public void SecondaryTimingRebuildsBothSidesOfTheJoin(double primaryPreutter,
            double targetPreutter, double targetOverlap) {
            var project = new UProject();
            Format.Ustx.AddDefaultExpressions(project);
            project.timeAxis.BuildSegments(project);
            var track = new UTrack();
            var part = new UVoicePart();
            var seed = Enumerable.Range(0, 2).Select(i => {
                var note = UNote.Create();
                note.position = i * 480;
                note.duration = note.ExtendedDuration = 480;
                var phone = new UPhoneme { Parent = note, position = note.position };
                // Seed the validated timing independently of a disk-backed singer.
                typeof(UPhoneme).GetProperty(nameof(UPhoneme.Duration)).SetValue(phone, 480);
                typeof(UPhoneme).GetProperty(nameof(UPhoneme.PositionMs)).SetValue(phone, i * 500.0);
                typeof(UPhoneme).GetProperty(nameof(UPhoneme.EndMs)).SetValue(phone, (i + 1) * 500.0);
                return phone;
            }).ToArray();
            UOto Oto(double preutter, double overlap) {
                var oto = UOto.OfDummy("a");
                oto.Preutter = preutter;
                oto.Overlap = overlap;
                return oto;
            }
            var primaryOtos = new[] { Oto(primaryPreutter, 10), Oto(primaryPreutter, 10) };
            var primary = UPhoneme.CreateRenderCopies(seed, primaryOtos, project, track, part);
            var originalEnvelopes = primary.Select(p => p.envelope.data.ToArray()).ToArray();
            var targetOtos = new[] { Oto(targetPreutter, targetOverlap), Oto(targetPreutter, targetOverlap) };
            var secondary = UPhoneme.CreateRenderCopies(primary, targetOtos, project, track, part);

            Assert.Equal(targetPreutter, secondary[1].preutter);
            Assert.Equal(targetOverlap, secondary[1].overlap);
            Assert.Equal(-targetPreutter, secondary[1].envelope.data[0].X);
            Assert.Equal(500 - targetPreutter + targetOverlap, secondary[0].envelope.data[4].X);
            Assert.Equal(secondary[1].PositionMs - secondary[1].preutter + secondary[1].overlap,
                secondary[0].PositionMs + secondary[0].envelope.data[4].X);
            // A shorter target must not inherit the primary start and need artificial left silence.
            Assert.Equal(0, targetPreutter + secondary[1].envelope.data[0].X);
            for (int i = 0; i < primary.Length; i++) {
                Assert.Equal(originalEnvelopes[i], primary[i].envelope.data);
                Assert.Same(primaryOtos[i], primary[i].oto);
            }
        }

        [Fact]
        public void PhraseEdgesRespectTheColorStartOnTheSharedTimeline() {
            var phone = new HiFiUtauPhone {
                PositionMs = 100,
                Envelope = [new(-20, 0), new(-10, 100), new(0, 100), new(80, 100), new(100, 0)],
            };
            var samples = HiFiUtauMath.BuildPhoneFadeEnvelope([phone], 220, 1000, 0);
            Assert.All(samples.Take(81), sample => Assert.Equal(0f, sample));
            Assert.Equal(1f, samples[90]);
            Assert.Equal(1f, samples[179]);
            Assert.Equal(0.05f, samples[199]);
            Assert.All(samples.Skip(200), sample => Assert.Equal(0f, sample));
        }

        [Theory]
        [InlineData(20, 20, 1, 15, 1)]
        [InlineData(20, 60, 2.0 / 3, 17.5, 1)]
        [InlineData(80, 20, 0.625, 12, 0.625)]
        [InlineData(80, 60, 7.0 / 24, 100.0 / 7, 0.625)]
        [InlineData(10, 10, 2, 15, 1)]
        public void OverlapUsesIndependentAttackAndRelease(double attackMs, double releaseMs,
            double expectedGain, double expectedFeature, double expectedAfterOverlap) {
            var previous = new HiFiUtauPhone {
                PositionMs = 0,
                Envelope = [new(0, 0), new(10, 100), new(20, 100),
                    new((float)(120 - releaseMs), 100), new(120, 0)],
            };
            var next = new HiFiUtauPhone {
                PositionMs = 100,
                Envelope = [new(0, 0), new((float)attackMs, 100), new((float)attackMs, 100),
                    new(120, 100), new(140, 0)],
            };
            var gains = HiFiUtauMath.BuildPhoneFadeEnvelope([previous, next], 240, 1000, 0);
            Assert.Equal(expectedGain, gains[110], 6);
            // A long attack still attenuates the phone after the preceding phone ends.
            Assert.Equal(expectedAfterOverlap, gains[150], 6);
            // A long release also starts before the overlap begins.
            Assert.Equal(releaseMs > 40 ? 40 / releaseMs : 1, gains[80], 6);

            var a = new float[1, 1, 21];
            var b = new float[1, 1, 21];
            for (int t = 0; t < 21; t++) {
                a[0, 0, t] = 10;
                b[0, 0, t] = 20;
            }
            var blended = HiFiUtauModel.CrossFadeFeat(a, b, previous, next, 100, 1);
            Assert.Equal(expectedFeature, blended[0, 0, 10], 5);
            Assert.Equal(10f, a[0, 0, 10]);
            Assert.Equal(20f, b[0, 0, 10]);
        }
    }
}

import pandas as pd

DATA_PATH = "dataset/flights.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df = df[
    (df["CANCELLED"] == 0) &
    (df["ARR_DELAY"].notna()) &
    (df["ARR_DELAY"] > 15)
].copy()

delay = df["ARR_DELAY"]

print("\n" + "=" * 50)
print("AEROALERT DELAY DISTRIBUTION")
print("=" * 50)

print(f"\nDelayed flights: {len(delay):,}")

print(f"\nMean delay   : {delay.mean():.1f} minutes")
print(f"Median delay : {delay.median():.1f} minutes")
print(f"25th percentile : {delay.quantile(0.25):.1f} minutes")
print(f"75th percentile : {delay.quantile(0.75):.1f} minutes")
print(f"90th percentile : {delay.quantile(0.90):.1f} minutes")
print(f"95th percentile : {delay.quantile(0.95):.1f} minutes")
print(f"Maximum delay : {delay.max():.1f} minutes")
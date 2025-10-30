
#29-10-25
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "mushrooms.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "uciml/mushroom-classification",
    file_path,
)

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.columns.tolist())
print(df.head())
print(df['class'].value_counts())

df['class'].map({'e':0, 'p':1})

# Download latest version
path = kagglehub.dataset_download("uciml/mushroom-classification")

print("Path to dataset files:", path)


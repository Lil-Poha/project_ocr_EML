import re, unicodedata, argparse
from pathlib import Path
import cv2, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .charset import Charset

RAW = Path("data/raw")
OUT = Path("data/processed")
SEED = 42

COL_FILE, COL_TEXT = "image_path", "label"
IMG_EXT = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
IMG_DIRS = [RAW / "dataset", RAW / "teacher"]   # где искать картинки


REPLACE = {
    "“": '"', "”": '"', "–": "-", "—": "-",   # пример, дополни по факту
}


def clean(t) -> str:
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[\u200b\u200c\u200d\ufeff\xa0]", " ", t)
    t = "".join(REPLACE.get(c, c) for c in t)
    return re.sub(r"\s+", " ", t).strip()


def find_image(rel: str):
    for base in IMG_DIRS:
        for ext in IMG_EXT + [""]:
            p = base / (rel + ext)
            if p.is_file():
                return p
    return None


def collect() -> pd.DataFrame:
    rows = []
    missing_rows = []          # перед циклом
    for xlsx in sorted((RAW / "teacher").glob("*.xlsx")):
        df = pd.read_excel(xlsx, dtype=str)          # dtype=str: «446» и «1927» остаются текстом
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=[COL_FILE, COL_TEXT])  # пустые ячейки иначе станут строкой "nan"
        for _, r in df.iterrows():
            rel = r[COL_FILE].strip().replace("\\", "/")
            p = find_image(rel)
            
            if p is None:
                missing_rows.append(dict(rel=rel, text=r[COL_TEXT]))
                continue
            rows.append(dict(path=str(p), text=r[COL_TEXT],
                             source=xlsx.stem, author=rel.split("/")[0]))   # author = папка (32_1)
        m = pd.DataFrame(missing_rows)                      # после цикла
        if len(m):
            m["folder"] = m.rel.str.split("/").str[0]
            # print(m.folder.value_counts().to_string())      # сколько пропусков по папкам
            m.to_csv(OUT / "missing.csv", index=False)
    print(f"Не найденные пути в {OUT}/missing.csv")
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(clean)
    n0 = len(df)
    df = df[df.text.str.len() > 0]
    df = df[df.path.map(lambda p: Path(p).exists())]
    df = df[df.path.map(lambda p: cv2.imread(p, cv2.IMREAD_GRAYSCALE) is not None)]
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    print(f"строк: {n0} -> {len(df)} после чистки")
    return df


def split(df: pd.DataFrame, val=0.15, test=0.15) -> pd.DataFrame:
    """Сплит по авторам/папкам, общий для всех моделей (seed фиксирован)."""
    df = df.copy()
    g1 = GroupShuffleSplit(1, test_size=val + test, random_state=SEED)
    tr, rest = next(g1.split(df, groups=df.author))
    rest_df = df.iloc[rest]
    g2 = GroupShuffleSplit(1, test_size=test / (val + test), random_state=SEED)
    v, t = next(g2.split(rest_df, groups=rest_df.author))
    df["split"] = "train"
    df.loc[rest_df.index[v], "split"] = "val"
    df.loc[rest_df.index[t], "split"] = "test"
    print(df.split.value_counts().to_string())
    return df


def stats(df: pd.DataFrame, height=64):
    print("длины текстов:\n", df.text.str.len().describe(percentiles=[.5, .95, .99]))
    ws = []
    for p in df.path.sample(min(len(df), 2000), random_state=0):
        h, w = cv2.imread(p, cv2.IMREAD_GRAYSCALE).shape
        ws.append(round(w * height / h))
    print(f"ширина после ресайза до H={height}:\n",
          pd.Series(ws).describe(percentiles=[.5, .95, .99]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=64)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = split(validate(collect()))
    df.to_csv(OUT / "labels.csv", index=False)

    cs = Charset.from_texts(df.text)           # по всему датасету
    cs.save(OUT / "charset.json")
    print(f"charset: {len(cs)} классов с blank")
    counts = pd.Series(list("".join(df.text))).value_counts()
    print("редкие символы (<5):", counts[counts < 5].to_dict())
    stats(df, args.height)

if __name__ == "__main__":
    main()
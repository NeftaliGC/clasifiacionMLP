"""
EDA script: eda.py

Resumen:
- Provee la función `run_eda(csv_path, output_dir='eda_output', target_col='Col17', sample_frac=None, top_n_categories=10, save_plots=True)`
- Genera resúmenes numéricos y categóricos y la matriz de correlación.
- Guarda **solo** estos plots: histogramas (uno por variable numérica), distribución del target y heatmap de correlaciones.
- Crea un reporte en Markdown `eda_report.md` que incluye: metadatos, los mismos summaries (numeric_summary, categorical_summary, numeric_correlations), las imágenes seleccionadas, y una interpretación automática (breve) basada en los summaries.
- NO realiza ningún preprocesamiento sobre los datos — solo análisis exploratorio.
- Está pensado para ser importado desde un `main.py`.

Dependencias:
- pandas, numpy, matplotlib, seaborn

Ejemplo de uso desde main.py:
```py
from eda import run_eda
run_eda('data.csv', output_dir='eda_out', target_col='Col17')
```
"""

from pathlib import Path
import warnings
import argparse
import textwrap
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _is_categorical_series(s: pd.Series) -> bool:
    """Detecta si una columna es categórica.

    Reglas usadas:
        - Si el dtype es object/string => categórica
        - O si más del 50% de los valores siguen el patrón 'cat_' => categórica
    """
    if pd.api.types.is_object_dtype(s.dtype):
        return True
    # check pattern 'cat_'
    non_null = s.dropna().astype(str)
    if len(non_null) == 0:
        return False
    frac_cat_prefix = non_null.str.startswith('cat_').mean()
    return frac_cat_prefix >= 0.5


def _numeric_summary(df_num: pd.DataFrame) -> pd.DataFrame:
    desc = df_num.describe().T
    desc = desc.rename(columns={
        '25%': 'q1', '50%': 'median', '75%': 'q3'
    })
    desc['skew'] = df_num.skew()
    desc['kurtosis'] = df_num.kurtosis()
    desc['n_missing'] = df_num.isna().sum()
    desc['n_unique'] = df_num.nunique()
    return desc


def _categorical_summary(df_cat: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for col in df_cat.columns:
        s = df_cat[col]
        n_missing = s.isna().sum()
        n_unique = s.nunique(dropna=True)
        top = s.value_counts(dropna=True).head(top_n).to_dict()
        mode = s.mode().iloc[0] if not s.mode().empty else np.nan
        rows.append({
            'column': col,
            'n_missing': n_missing,
            'n_unique': n_unique,
            'mode': mode,
            'top_values': top
        })
    return pd.DataFrame(rows).set_index('column')


def _simple_interpret_numeric_row(name: str, row: pd.Series) -> str:
    parts = []
    # skew
    skew = row.get('skew', 0)
    if pd.notna(skew):
        if skew > 1:
            parts.append('distribución con cola positiva (sesgo > 1)')
        elif skew > 0.5:
            parts.append('ligeramente sesgada a la derecha')
        elif skew < -1:
            parts.append('distribución con cola negativa fuerte (sesgo < -1)')
        elif skew < -0.5:
            parts.append('ligeramente sesgada a la izquierda')
        else:
            parts.append('aproximadamente simétrica')
    # unique
    n_unique = row.get('n_unique', None)
    if pd.notna(n_unique):
        if n_unique <= 10:
            parts.append(f'poca cardinalidad ({int(n_unique)} valores únicos)')
        else:
            parts.append(f'alta cardinalidad ({int(n_unique)} valores únicos)')
    # zeros
    q1 = row.get('q1', np.nan)
    median = row.get('median', np.nan)
    if pd.notna(q1) and q1 == 0:
        parts.append('inflación en ceros (Q1 = 0)')

    return f"{name}: " + "; ".join(parts) + '.'


def _generate_markdown(out: Path, meta: dict, num_summary: pd.DataFrame, cat_summary: pd.DataFrame, corr: pd.DataFrame, plots_dir: Path, target_col: str):
    md_path = out / 'eda_report.md'
    with md_path.open('w', encoding='utf8') as f:
        f.write('# EDA Report\n')
        f.write(f'Date: {pd.Timestamp.now()}\n')
        f.write('## Metadata\n')
        f.write('| Key | Value |\n')
        f.write('|---:|:---|\n')
        f.write(f"| n_rows | {meta['n_rows']} |\n")
        f.write(f"| n_columns | {meta['n_columns']} |\n")
        f.write(f"| numeric_cols | {', '.join(meta['numeric_cols'])} |\n")
        f.write(f"| categorical_cols | {', '.join(meta['categorical_cols'])} |\n")
        f.write(f"| target_col | {meta['target_col']} |\n")
        f.write('---\n')

        # Numeric summary table
        f.write('## Numeric summary\n')
        f.write('')
        if not num_summary.empty:
            f.write(num_summary.reset_index().to_markdown(index=False))
            f.write('\n')
        else:
            f.write('No numeric columns detected.\n')

        # Categorical summary
        f.write('## Categorical summary\n')
        f.write('')
        if not cat_summary.empty:
            # convert top_values dict to JSON string for readability
            cat_csv = cat_summary.copy()
            cat_csv['top_values'] = cat_csv['top_values'].apply(lambda d: json.dumps(d, ensure_ascii=False))
            f.write(cat_csv.reset_index().to_markdown(index=False))
            f.write('\n')
        else:
            f.write('No categorical columns detected.')

        # Correlation matrix
        f.write('## Numeric correlations (pearson)\n')
        f.write('')
        if corr is not None and not corr.empty:
            f.write(corr.round(3).reset_index().to_markdown(index=False))
            f.write('\n')
        else:
            f.write('No correlations to show.')

        # Plots
        f.write('## Plots\n')
        f.write('')
        f.write('### Correlation heatmap\n')
        heatmap_path = plots_dir / 'correlation_heatmap.png'
        if heatmap_path.exists():
            f.write(f'![](./{heatmap_path.relative_to(out)})')
        else:
            f.write('Heatmap not found.')

        f.write('### Histograms (numeric features)\n')
        for col in meta['numeric_cols']:
            img = plots_dir / f'hist_{col}.png'
            if img.exists():
                f.write(f'#### {col}\n')
                f.write(f'![](./{img.relative_to(out)})\n')

        f.write('### Target distribution\n')
        img_t = plots_dir / f'target_distribution_{target_col}.png'
        if img_t.exists():
            f.write(f'![](./{img_t.relative_to(out)})\n')

        # Automated interpretation
        f.write('## Interpretación automática (resumen)\n')
        f.write('\n')
        f.write('## Resumen general\n')
        # target balance
        if 'target_counts' in meta:
            tc = meta['target_counts']
            mx = max(tc.values())
            mn = min(tc.values())
            ratio = mx / mn if mn > 0 else float('inf')
            if ratio < 2:
                f.write('- El objetivo parece relativamente balanceado entre clases.')
            else:
                f.write('- El objetivo presenta clases desbalanceadas.')
        else:
            f.write('- No se incluyeron conteos del objetivo en los metadatos.')

        f.write('\n')
        f.write('### Numeric features (observaciones rápidas)\n')
        if not num_summary.empty:
            for name, row in num_summary.iterrows():
                f.write(f'- {_simple_interpret_numeric_row(str(name), row)}\n')
        else:
            f.write('- Sin columnas numéricas.\n')

        f.write('\n')
        f.write('### Categorical features (observaciones rápidas)\n')
        if not cat_summary.empty:
            for name, row in cat_summary.iterrows():
                top = row['top_values']
                # top is a dict; create brief phrase
                if isinstance(top, dict):
                    items = list(top.items())[:3]
                    items_str = ', '.join([f'{k}: {v}' for k, v in items])
                else:
                    items_str = str(top)
                dominance = ''
                total = sum(top.values()) if isinstance(top, dict) else None
                if isinstance(top, dict) and len(top) > 0:
                    most_freq = max(top.values())
                    if total and most_freq / total > 0.8:
                        dominance = ' (una categoría domina >80%)'
                f.write(f'- {name}: top values: {items_str}.{dominance}\n')
        else:
            f.write('- Sin columnas categóricas.')

        f.write('\n')
        f.write('### Correlaciones notables\n')
        if corr is not None and not corr.empty:
            # list pairs with abs(corr) >= 0.4 (except self)
            pairs = []
            cols = corr.columns.tolist()
            for i, a in enumerate(cols):
                for j, b in enumerate(cols):
                    if j <= i:
                        continue
                    val = corr.loc[a, b]
                    if np.isfinite(float(val)) and abs(val) >= 0.4: # type: ignore
                        pairs.append((a, b, val))
            if pairs:
                for a, b, val in pairs:
                    f.write(f'- {a} vs {b}: r = {val:.2f}')
            else:
                f.write('- No se encontraron correlaciones fuertes (|r| >= 0.4).')
        else:
            f.write('- Sin matriz de correlación.')

    return md_path


def run_eda(csv_path: str,
            output_dir: str = 'eda_output',
            target_col: str = 'Col17',
            sample_frac: float | None = None,
            top_n_categories: int = 10,
            save_plots: bool = True,
            verbose: bool = True):
    """Ejecuta un análisis exploratorio y guarda resultados en output_dir.

    Este ajuste crea menos plots y produce un único reporte en Markdown.
    """
    csv_path_obj = Path(csv_path)
    out = Path(output_dir)
    _ensure_dir(out)
    plots_dir = out / 'plots'
    if save_plots:
        _ensure_dir(plots_dir)

    if verbose:
        print(f"Leyendo CSV: {csv_path_obj}")
    df = pd.read_csv(csv_path_obj)

    if sample_frac is not None and 0 < sample_frac < 1:
        df_sample = df.sample(frac=sample_frac, random_state=42)
    else:
        df_sample = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' no está en las columnas del CSV")

    # Detectar tipos
    col_types = {}
    numeric_cols = []
    categorical_cols = []
    for c in df.columns:
        if _is_categorical_series(df[c]):
            categorical_cols.append(c)
            col_types[c] = 'categorical'
        else:
            # intentar convertir a numérico
            coerced = pd.to_numeric(df[c], errors='coerce')
            num_non_na = coerced.notna().sum()
            orig_non_na = df[c].dropna().shape[0]
            # si la conversión conserva la mayoría -> numérica
            if orig_non_na > 0 and (num_non_na / orig_non_na) >= 0.9:
                numeric_cols.append(c)
                col_types[c] = 'numeric'
            else:
                categorical_cols.append(c)
                col_types[c] = 'categorical'

    # Forcing the target to categorical
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col not in categorical_cols:
        categorical_cols.append(target_col)
    col_types[target_col] = 'categorical'

    # Basic metadata
    meta = {
        'n_rows': int(df.shape[0]),
        'n_columns': int(df.shape[1]),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'target_col': target_col
    }

    # Missing values
    miss = df.isna().sum().rename('n_missing').to_frame()
    miss['pct_missing'] = (miss['n_missing'] / len(df)) * 100
    miss.to_csv(out / 'missing_values.csv')

    # Numeric summary
    df_num = df[numeric_cols].apply(pd.to_numeric, errors='coerce') if numeric_cols else pd.DataFrame()
    num_summary = _numeric_summary(df_num) if not df_num.empty else pd.DataFrame()
    num_summary.to_csv(out / 'numeric_summary.csv')

    # Categorical summary
    df_cat = df[categorical_cols].astype(object) if categorical_cols else pd.DataFrame()
    cat_summary = _categorical_summary(df_cat, top_n=top_n_categories)
    # expand top_values into JSON-like string for CSV
    cat_summary_csv = cat_summary.copy()
    cat_summary_csv['top_values'] = cat_summary_csv['top_values'].apply(lambda d: str(d))
    cat_summary_csv.to_csv(out / 'categorical_summary.csv')

    results = {
        'meta': meta,
        'missing_values_csv': str(out / 'missing_values.csv'),
        'numeric_summary_csv': str(out / 'numeric_summary.csv'),
        'categorical_summary_csv': str(out / 'categorical_summary.csv')
    }

    # Correlation matrix (numeric only)
    corr = pd.DataFrame()
    if not df_num.empty:
        corr = df_num.corr()
        corr.to_csv(out / 'numeric_correlations.csv')
        results['numeric_correlations_csv'] = str(out / 'numeric_correlations.csv')

        if save_plots:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', square=True, cbar_kws={'shrink': .8})
            plt.title('Correlation matrix (numeric)')
            plt.tight_layout()
            plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=150)
            plt.close()
            results['correlation_heatmap'] = str(plots_dir / 'correlation_heatmap.png')

    # Plots numeric: histograms ONLY
    for col in numeric_cols:
        s = pd.to_numeric(df_sample[col], errors='coerce')
        if s.dropna().empty:
            continue
        if save_plots:
            plt.figure()
            sns.histplot(s.dropna(), kde=True)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(plots_dir / f'hist_{col}.png', dpi=150)
            plt.close()

        # outlier detection using IQR (kept for summary)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = s[(s < lower) | (s > upper)].shape[0]
        num_summary.loc[col, 'iqr'] = iqr
        num_summary.loc[col, 'n_outliers_iqr'] = n_outliers

    # Target distribution (plot only)
    target_counts = df[target_col].value_counts(dropna=False)
    meta['target_counts'] = target_counts.to_dict()
    target_counts.to_frame(name='count').to_csv(out / f'target_distribution_{target_col}.csv')
    if save_plots:
        plt.figure()
        sns.barplot(x=target_counts.index.astype(str), y=target_counts.values)
        plt.title(f'Target distribution: {target_col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / f'target_distribution_{target_col}.png', dpi=150)
        plt.close()

    # Save updated numeric_summary with IQR/outliers if collected
    if not num_summary.empty:
        num_summary.to_csv(out / 'numeric_summary_with_outliers.csv')
        results['numeric_summary_with_outliers'] = str(out / 'numeric_summary_with_outliers.csv')

    # Generate markdown report
    md_path = _generate_markdown(out, meta, num_summary, cat_summary, corr, plots_dir, target_col)
    results['report_md'] = str(md_path)

    if verbose:
        print('\nEDA finalizado. Archivos guardados en:', out)
        print('Resumen rápido:')
        print(' - Filas:', meta['n_rows'])
        print(' - Columnas:', meta['n_columns'])
        print(' - Numéricas:', len(numeric_cols))
        print(' - Categóricas:', len(categorical_cols))

        print('\nPrincipales archivos:')
        for k, v in results.items():
            # Formatear salida
            if k == 'meta':
                print(f' - {k}:')
                for mk, mv in v.items():
                    print(f'    - {mk}: {mv}')
            else:
                print(f' - {k}: {v}')

    return results


# CLI: para ejecutar directamente desde la terminal si el usuario lo prefiere
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDA script (reduced plots + markdown report)')
    parser.add_argument('csv', help='Path to CSV file')
    parser.add_argument('--out', '-o', default='eda_output', help='Output directory')
    parser.add_argument('--target', '-t', default='Col17', help='Target column name')
    parser.add_argument('--sample', type=float, default=None, help='Sample fraction (0-1) for quicker plots')
    args = parser.parse_args()

    run_eda(args.csv, output_dir=args.out, target_col=args.target, sample_frac=args.sample)
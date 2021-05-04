from process_data import get_processed_data

def write_cache(path):
    """
    filter highly variable genes and write adata to cache
    """
    adata = get_processed_data(path)
    print(f'after transpose: {adata.shape}')
    adata.write_h5ad('./cache/processed-adata-cache.h5ad')

if __name__ == "__main__":
    write_cache('./data/')
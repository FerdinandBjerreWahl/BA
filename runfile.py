import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')
    output_notebook, _ = ep.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})

    # Print the output of each executed cell
    for cell in output_notebook.cells:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    print(output['text'])

    # Optionally, you can also save the executed notebook if needed
    output_path = os.path.splitext(notebook_path)[0] + '_executed.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(output_notebook, f)

if __name__ == '__main__':
    notebooks = [
        'Covarians_and_mean_test.ipynb',
        # Add more notebook paths as needed
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    for notebook in notebooks:
        notebook_path = os.path.join(base_dir, notebook)
        run_notebook(notebook_path)





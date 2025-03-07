import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium", app_title="Semantic Search in the browser with Transformers.js and Marimo")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Semantic Search in the browser with Transformers.js and Marimo

        *By [Harry Vangberg](https://harry.vangberg.name)*

        [Marimo](https://marimo.io/) is a next-gen reactive Python notebook. Notebooks can run in the browser, thanks to [Pyodide](https://pyodide.org/)-powered [WASM export](https://docs.marimo.io/guides/wasm/). In this notebook I will walk through using [Transformers.js](https://huggingface.co/docs/transformers.js/en/index) from [HuggingFace](https://huggingface.co/) in combination with Marimo to build a semantic search solution that runs entirely in the browser.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Sentences

        The input sentences that we want to search:
        """
    )
    return


@app.cell
def _():
    sentences = [
        "The giraffe stretched its long neck.",
        "The tiger prowled through the jungle.",
        "Boats float on the water.",
        "The monkey swung from tree to tree.",
        "Dogs bark at passing strangers.",
        "The dolphin jumped out of the ocean.",
        "Trains move along steel tracks.",
        "The owl hunted during the night.",
        "Bicycles require pedaling to move.",
        "The penguin waddled across the ice.",
    ]
    return (sentences,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Embeddings

        To generate vector embeddings, we will use [Transformers.js.py](https://github.com/whitphx/transformers.js.py), a package that allows us to use Transformers.js from Pyodide. We use the feature extraction pipeline with the [all-MiniLM-l6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) sentence transformer model.
        """
    )
    return


@app.cell
async def _(sentences):
    import micropip

    await micropip.install("transformers-js-py")
    from transformers_js_py import import_transformers_js

    transformers = await import_transformers_js()
    pipeline = transformers.pipeline


    async def get_embeddings(inputs):
        extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2")
        output = await extractor(inputs, {"pooling": "mean", "normalize": True})
        return output.to_numpy()


    embeddings = await get_embeddings(sentences)
    return (
        embeddings,
        get_embeddings,
        import_transformers_js,
        micropip,
        pipeline,
        transformers,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Search

        Finally, we have arrived. Whenever the text input below is updated, an embedding is generated from the search query. We use [scikit-learn](https://scikit-learn.org/) to perform [nearest neighbor search](https://en.wikipedia.org/wiki/Nearest_neighbor_search) and return the three nearest sentences. Try to change the search query below to "Animals".
        """
    )
    return


@app.cell
def _(mo):
    search = mo.ui.text("Vehicles", placeholder="Search", debounce=False)
    search
    return (search,)


@app.cell
async def _(embeddings, get_embeddings, search, sentences):
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3, algorithm="auto")
    nn.fit(embeddings)

    search_embeddings = await get_embeddings(search.value)

    distances, indices = nn.kneighbors(search_embeddings[0].reshape(1, -1))

    nearest_sentences = [sentences[idx] for idx in indices[0]]
    nearest_sentences
    return (
        NearestNeighbors,
        distances,
        indices,
        nearest_sentences,
        nn,
        search_embeddings,
    )


if __name__ == "__main__":
    app.run()

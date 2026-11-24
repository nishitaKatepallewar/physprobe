# Hosting the explorer

The hosted explorer is read-only. It shows a **results bundle**: probe, ablation and attention
tables, figures, dataset previews and precomputed shapes, without activation arrays and without
the model. Visitors can open their own bundles with the upload box on the Overview page.

## 1. Create or refresh the bundle

On the machine that holds the runs:

```bash
python -m physprobe export --runs runs --data data --out results
```

This replaces `results/`. Shapes (condition-mean trajectories, video projections and shape
statistics for the Shape explorer) are computed from the activation caches during export, so
export must run where `runs/*/acts/pooled.npy` exist. `--labels direction_deg speed` limits
which variables are exported; the default is each dataset's design factors. A typical bundle
is 2–20 MB. Commit `results/` (the `.gitignore` keeps its `.npz` files).

`--zip` additionally writes `results.zip` for sharing or uploading; it is not committed.

## 2. Streamlit Community Cloud (free)

1. Push the repository to GitHub (public, or private with Streamlit's GitHub access).
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app → Deploy a public
   app from GitHub**.
3. Repository: this one. Branch: `main`. Main file path: `hosted/streamlit_app.py`.
4. Under **Advanced settings**, pick Python 3.11 or newer. Deploy.

Streamlit reads `hosted/requirements.txt` (next to the entry point; no torch, so the build
takes a few minutes) and the theme in `.streamlit/config.toml`. Every push to `main` redeploys.
Apps on the free tier sleep after a period without visitors and wake on the next visit (about
half a minute); memory is limited to about 1 GB, so bundles of a few hundred MB are not
suitable there.

## 3. Hugging Face Spaces (alternative)

Create a Space with the **Docker** SDK and add this `Dockerfile` to the repository root:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r hosted/requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "hosted/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

The Space's `README.md` front matter needs `sdk: docker` and `app_port: 8501`.

## 4. Locally, in hosted mode

```bash
pip install -r hosted/requirements.txt
streamlit run hosted/streamlit_app.py
```

or, with the package installed, `python -m physprobe gui --runs results --data results/datasets --hosted`.

## What hosted mode changes

- Pages that start computations (new probe runs, new datasets, video import) are hidden;
  each page shows the command to run locally instead.
- Uploaded bundles are unpacked into a temporary folder per session, checked for paths that
  escape the folder, and limited to 500 MB unpacked. They are not stored after the session ends.
- The Shape explorer uses the precomputed shapes; subsets of videos (filters on other factors)
  need a live activation cache and are available only locally.

Vercel and similar serverless hosts are not suitable: Streamlit keeps a long-lived WebSocket
connection per visitor, which serverless functions do not support for long.

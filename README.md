
# Determinate Beam – Shear & Moment Calculator (Web)

A Streamlit web app port of your Tkinter tool. It lets you configure beam type, length, supports, and add loads (point loads, point moments, UDLs, and linear/triangular loads) then computes and plots the shear and moment diagrams. It also identifies extrema and the first zero-shear location and lets you export the computed arrays as CSV.

## How to run locally

1. Ensure you have Python 3.9+ installed.
2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

5. Your browser will open automatically (or visit http://localhost:8501). Use the sidebar to add loads and click **Compute & Plot**.

## Notes
- This port fixes a small bug from the original: in `shear_and_moment`, the cantilever shear jump now correctly uses `self.beam_type == "Cantilever"` inside the BeamModel context.
- The schematic drawing is simplified versus the Tkinter version but retains all key information.
- CSV export is optional (enable the checkbox after computing).

## Deploying online (optional)
- You can deploy to Streamlit Community Cloud or any platform that supports Streamlit. On Streamlit Cloud, upload `app.py` and `requirements.txt` to a public repo and deploy.

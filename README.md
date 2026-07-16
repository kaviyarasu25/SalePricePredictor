# London Residential Property Price Predictor

A Streamlit web app that wraps the Random Forest model from the dissertation
"Machine Learning-Based Residential Property Price Prediction for London."
Users enter property characteristics and get an estimated market value.

## What's in here

```
property-price-app/
├── app.py              # the Streamlit app (form + preprocessing + prediction)
├── requirements.txt    # Python dependencies
├── model/
│   ├── .gitkeep
│   ├── random_forest_model.pkl   # <-- YOU ADD THIS (your trained model, ~1.7GB)
│   └── model_features.pkl        # <-- YOU ADD THIS (already provided)
└── README.md
```

The app has been fully tested end-to-end against a stand-in model with the
same 24-feature schema as yours — the preprocessing pipeline, one-hot
encoding, feature ordering, and market index lookup are all verified correct.
You just need to drop your real files into `model/`.

## 1. Run it locally first

```bash
cd property-price-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy your two files into the `model/` folder:
- `random_forest_model.pkl` (your trained model — the compressed ~1.7GB one)
- `model_features.pkl` (already provided)

Then run:

```bash
streamlit run app.py
```

It'll open at `http://localhost:8501`. Try a few predictions and sanity-check
the numbers against what you'd expect (e.g. compare to actual sold prices in
your test set for similar properties).

**Note on the model file:** with scikit-learn 1.8.0 / joblib 1.5.3 as pinned
in `requirements.txt`. If your model was trained with different versions,
either match them locally, or re-save the model with your current versions
before proceeding (a version mismatch can throw errors or silently degrade
predictions).

## 2. Deploy to Hugging Face Spaces (free, handles the 1.7GB file)

1. Create a free account at https://huggingface.co if you don't have one.
2. Create a new Space: **New Space** → choose **Streamlit** as the SDK →
   name it (e.g. `london-property-price-predictor`) → set visibility to
   **Public** so anyone can test it.
3. Clone the Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   ```
4. Copy `app.py`, `requirements.txt`, and the `model/` folder (with your
   real `.pkl` files inside) into this cloned repo.
5. Because the model file is large, set up Git LFS **before** committing it:
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git add .gitattributes
   ```
6. Commit and push:
   ```bash
   git add .
   git commit -m "Add property price predictor app and model"
   git push
   ```
7. Hugging Face will automatically build and deploy the Space. Give it a
   few minutes (large file upload + first build). Your app will be live at:
   ```
   https://huggingface.co/spaces/<your-username>/<space-name>
   ```
   That's the link you can share with anyone to test it — no installation
   required on their end.

**If the push is slow or times out:** git-lfs uploads of ~1.7GB can take a
while on a slow connection. Let it run; you can check upload progress with
`git lfs status` / by watching the Space's "Files" tab build logs online.

## How the app works

1. User fills in a form: location (lat/long), property type, construction
   period, floor area, room count, energy rating, floor level, and a
   valuation year.
2. The app reproduces the exact feature engineering from the dissertation:
   - `distance_to_centre_km` computed via geodesic distance to Charing Cross
   - `property_category` / `construction_period` one-hot encoded with the
     same baseline categories dropped during training (Bungalow / 1900-1949)
   - `energy_rating_score` mapped A→7 ... G→1
   - `market_index` looked up from the exact year→index table computed
     during training (median price growth relative to 1995 baseline)
   - floor level missingness handled the same way (0 + missing flag)
3. The resulting feature vector is reindexed to match `model_features.pkl`
   exactly, then passed to the model.
4. The model predicts on the log-price scale (matching training); the app
   exponentiates it back to pounds sterling for display.

## Known limitations (carried over from the dissertation, Section 7.13)

- Predictions for unusually expensive/luxury properties carry more
  uncertainty (thinly represented in training data).
- The model was trained on data through 2026 — predictions for later years
  fall outside the trained market index range.
- Valid only within the 12 London boroughs studied; coordinates outside
  Greater London are not meaningful inputs even though the form won't stop you.

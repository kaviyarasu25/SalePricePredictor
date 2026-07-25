# London Residential Property Price Predictor

This project is a Streamlit web application that predicts residential property prices in London using a trained Random Forest model developed as part of my MSc dissertation.

The application allows users to enter property details and returns an estimated property price based on the trained machine learning model.

## Project Files

```
SalePricePredictor-main/
│── app.py
│── requirements.txt
│── model/
│   ├── random_forest_model.pkl
│   └── model_features.pkl
└── README.md
```

## Running the Application

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Make sure the trained model files are placed inside the `model` folder:

- `random_forest_model.pkl`
- `model_features.pkl`

3. Start the application:

```bash
streamlit run app.py
```

The application will open in your browser, where you can enter property details and generate a predicted property price.

## How It Works

The application performs the same preprocessing steps used during model training before making a prediction. These include:

- Property type encoding
- Energy rating conversion
- Market index lookup
- Distance to Central London calculation
- Feature ordering to match the trained model

The processed data is then passed to the Random Forest model to generate the final price prediction.

## Notes

- This application is intended for properties within London.
- Predictions are based on the data used to train the model and may be less accurate for unusual or high-value properties.
- The model was developed as part of an MSc dissertation on residential property price prediction.

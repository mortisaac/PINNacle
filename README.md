<img width="1060" height="476" alt="Screenshot 2025-10-19 142850" src="https://github.com/user-attachments/assets/03bf3f93-5d57-45e3-8f2c-b4be77d4bbe5" />

# This is PINN-acle
PINN-acle uses NNs and PINNs to extract asset Pricing and Volatility - it merges market and news sentiment data to present key features of listed assets. Then it uses an LLM to translate this analysis into clear and actionable insights targeted to consumers and traders.

## API keys:
You will need a Google AI studio API key and a News API key to go in the .secrets file in .streamlit

## How to use:
Run the following in the python terminal: python -m streamlit run app.py
Submit any ticker into the search bar in the top right
You will have the option to analyse existing pretrained models (^SPX, SPY, AAPL, NVDA)
Alternatively you can insert any ticker and click (Re)-Train
View the analysis via the dashbaord

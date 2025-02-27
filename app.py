if stock_symbol:
    prediction, future_dates, processed_data = predict_stock(stock_symbol)

    if prediction is not None and future_dates is not None:
        latest_price = processed_data['Close'].iloc[-1]
        
        # Display the latest stock price in large size
        st.markdown(f"<h2 style='color: green; text-align: center;'>Latest Price: ${latest_price:.2f}</h2>", unsafe_allow_html=True)

        st.write(f"Prediction for {stock_symbol}:")

        # Display the predicted prices in red
        for date, pred in zip(future_dates, prediction):
            predicted_price = latest_price * (1 + pred / 100)
            st.markdown(f"<h3 style='color: red;'>On {date.date()}: ${predicted_price:.2f} ({pred:.2f}%)</h3>", unsafe_allow_html=True)

        plot_stock_data(processed_data, future_dates, prediction)
    else:
        st.write("Not enough data to make a prediction.")

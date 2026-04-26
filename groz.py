
@app.post("/journal/review")
async def review_trade(ticker: str, thesis: str):
    try:
        ticker = ticker.upper()
        # 1. Grab technical data from Redis to give the AI context
        # This prevents the AI from hallucinating old prices
        tech_data = await get_from_cache(f"analysis:{ticker}")
        
        context_str = ""
        if tech_data:
            context_str = f"Current Price: {tech_data['price']}, RSI: {tech_data['rsi']}, Signal: {tech_data['rsi_signal']}"

        # 2. The System Prompt (The "Personality")
        system_msg = (
            "You are the Talos AI Auditor. You provide critical, data-driven feedback on trade ideas. "
            "Be concise, objective, and highlight technical contradictions. Use JSON format."
        )
        
        user_msg = f"Ticker: {ticker}. Market Context: {context_str}. User Thesis: {thesis}"

        # 3. High-speed Inference
        completion = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"}
        )

        return orjson.loads(completion.choices[0].message.content)

    except Exception as e:
        return {"error": str(e)}
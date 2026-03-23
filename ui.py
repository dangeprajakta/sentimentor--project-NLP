import streamlit as st
from sentimentator import analyze_sentiment

# --- THE MAIN STREAMLIT APP ---
def main():
    st.title("Sentimentator - Sentiment Analysis Tool")
    st.write("Enter text below to analyze its sentiment using AI.")
    
    # Text input
    text = st.text_input("Enter your text here:", "")
    
    # Model selection (optional)
    model_name = st.selectbox("Choose a model:", 
                              ["distilbert-base-uncased-finetuned-sst-2-english", 
                               "cardiffnlp/twitter-roberta-base-sentiment-latest"],
                              index=0)
    
    # Submit button
    if st.button("Analyze Sentiment"):
        if text.strip():
            with st.spinner("Analyzing..."):
                result = analyze_sentiment([text], model_name)
                if result:
                    result = result[0]  # Since we pass a list, get the first result
            if result:
                st.success("Analysis Complete!")
                st.write(f"**Input Text:** {text}")
                st.write(f"**Sentiment:** {result['label']}")
                st.write(f"**Confidence:** {result['score']:.4f}")
                
                # Visual indicator
                if result['label'] == 'POSITIVE':
                    st.markdown("😊 **Positive sentiment detected!**")
                elif result['label'] == 'NEGATIVE':
                    st.markdown("😞 **Negative sentiment detected!**")
                else:
                    st.markdown("😐 **Neutral sentiment detected!**")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
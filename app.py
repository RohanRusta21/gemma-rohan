# import os
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForCausalLM

# def main():
#     st.title("RohanGPT Chatbot")

#     # Load the tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
#     model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it")

#     # Initialize chat history
#     chat_history = []

#     # Display chat history
#     for message in chat_history:
#         with st.echo("above"):
#             st.write(message)

#     # Get user input
#     user_input = st.text_input("You:", "Write me a poem about Machine Learning.")

#     if st.button("Generate"):
#         # Generate response
#         input_ids = tokenizer(user_input, return_tensors="pt")
#         outputs = model.generate(**input_ids)
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Display response
#         st.text_area("Chatbot:", value=response, height=200)

# if __name__ == "__main__":
#     main()


#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


from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
login()

access_token = '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'
model_id = "google/gemma-7b"

tokenizer = AutoTokenizer.from_pretrained(model_id, access_token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_id, access_token=access_token)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))






# pip install -q transformers
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# checkpoint = "CohereForAI/aya-101"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# # Turkish to English translation
# tur_inputs = tokenizer.encode("Translate to English: Aya cok dilli bir dil modelidir.", return_tensors="pt")
# tur_outputs = aya_model.generate(tur_inputs, max_new_tokens=128)
# print(tokenizer.decode(tur_outputs[0]))
# # Aya is a multi-lingual language model

# # Q: Why are there so many languages in India?
# hin_inputs = tokenizer.encode("भारत में इतनी सारी भाषाएँ क्यों हैं?", return_tensors="pt")
# hin_outputs = aya_model.generate(hin_inputs, max_new_tokens=128)
# print(tokenizer.decode(hin_outputs[0]))
# Expected output: भारत में कई भाषाएँ हैं और विभिन्न भाषाओं के बोली जाने वाले लोग हैं। यह विभिन्नता भाषाई विविधता और सांस्कृतिक विविधता का परिणाम है। Translates to "India has many languages and people speaking different languages. This diversity is the result of linguistic diversity and cultural diversity."


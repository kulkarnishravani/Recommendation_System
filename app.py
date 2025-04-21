import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Sample Product Data
# -------------------------------

products = pd.read_csv("products_100.csv")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🛒 Product Recommendation System")
st.write("Get smart product suggestions based on what you’ve browsed or purchased!")

# Simulated user behavior inputs
selected_behavior = st.multiselect(
    "Select products you have browsed or purchased before:",
    options=products['name'].tolist()
)

# Function to generate recommendations
def recommend_products(selected_products):
    if not selected_products:
        return pd.DataFrame()

    user_profile = products[products['name'].isin(selected_products)]
    user_tags = " ".join(user_profile['tags'].tolist())

    # Add user profile as a pseudo-product for similarity comparison
    full_data = products.copy()
    full_data.loc[len(full_data)] = {
        "product_id": 999,
        "name": "User Profile",
        "category": "User",
        "tags": user_tags
    }

    # Vectorization
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(full_data['tags'])

    # Cosine Similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    full_data = full_data.iloc[:-1]  # remove user row

    full_data['similarity'] = similarity_matrix.flatten()

    # Filter out already seen products
    recommended = full_data[~full_data['name'].isin(selected_products)]
    recommended = recommended.sort_values(by='similarity', ascending=False).head(5)

    return recommended[['name', 'category', 'similarity']]

# Button to get recommendations
if st.button("Get Recommendations"):
    results = recommend_products(selected_behavior)
    if not results.empty:
        st.success("Based on your interests, we recommend:")
        st.dataframe(results)
    else:
        st.warning("Please select at least one product to get recommendations.")

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and data
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv(r"dataset.csv")
df.dropna(inplace=True)

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox("Go to", ["🔮 Predict Completion"])

categorical_features = ['CourseCategory', 'DeviceType']
numerical_features = ['TimeSpentOnCourse', 'NumberOfVideosWatched', 'NumberOfQuizzesTaken', 'QuizScores', 'CompletionRate']
target = 'CourseCompletion'

# ---------------------- PAGE: PREDICTION ----------------------
if page == "🔮 Predict Completion":
    st.title("🎓 Online Learning Effectiveness Predictor")
    st.markdown("Predict if a student will complete an online course based on their engagement data.")

    # Feature 1 - Course Category
    course_category = st.selectbox("📘 Course Category", ["Data Science", "Arts", "Commerce", "Technology"])
    category_dict = {"Data Science": 0, "Arts": 1, "Commerce": 2, "Technology": 3}
    course_category_encoded = category_dict[course_category]

    # Feature 2 - Device Type
    device_type = st.selectbox("💻 Device Type", ["Mobile Phone", "Laptop"])
    device_dict = {"Mobile Phone": 0, "Laptop": 1}
    device_encoded = device_dict[device_type]

    # Feature 3 to 7 - Primary engagement metrics
    time_spent = st.slider("⏱️ Time Spent on Course (hours)", 0.0, 100.0, 20.0)
    videos_watched = st.number_input("🎥 Number of Videos Watched", 0, 100, 10)
    quizzes_taken = st.number_input("📝 Number of Quizzes Taken", 0, 50, 5)
    quiz_scores = st.slider("📊 Average Quiz Score (%)", 0.0, 100.0, 75.0)
    completion_rate = st.slider("📈 Completion Rate (%)", 0.0, 100.0, 60.0)

    # Additional Features
    with st.expander("➕ Additional Features (Auto-engagement / Feedback etc.)"):
        forum_participation = st.number_input("💬 Forum Participation Count", min_value=0, max_value=100, value=5)
        peer_interaction = st.slider("🤝 Peer Interaction Score (0-100)", 0, 100, 50)
        feedback_given = st.number_input("🗒️ Feedback Given Count", min_value=0, max_value=100, value=3)
        reminders_clicked = st.number_input("🔔 Reminders Clicked", min_value=0, max_value=100, value=2)
        support_usage = st.number_input("🆘 Support Ticket Usage Count", min_value=0, max_value=100, value=1)

    # Combine input
    input_data = np.array([[course_category_encoded, device_encoded, time_spent, videos_watched, quizzes_taken,
                            quiz_scores, completion_rate, forum_participation, peer_interaction,
                            feedback_given, reminders_clicked, support_usage]])

    # Engagement threshold check
    max_possible = np.array([3, 1, 100, 100, 50, 100, 100, 100, 100, 100, 100, 100])
    engagement_score = np.sum(input_data[0][2:])
    max_engagement = np.sum(max_possible[2:])
    engagement_percent = (engagement_score / max_engagement) * 100

    if st.button("🚀 Predict"):
        if engagement_percent < 30:
            st.warning(f"⚠️ Engagement level is very low ({engagement_percent:.1f}%). Encourage more activity.")
        else:
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.success("✅ The student is likely to complete the course. Keep it up! 🚀")
            else:
                st.error("❌ The student may not complete the course. More effort and engagement is recommended.")

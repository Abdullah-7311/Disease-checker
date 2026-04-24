
⚕
MedDiag Pro
AI Medical Diagnostic System
Version 2.0  —  Technical Documentation

56
Diseases	130+
Symptoms	200
RF Trees

Python  ·  CustomTkinter  ·  Scikit-learn  ·  Random Forest
Prepared: April 24, 2026
 
1.  Executive Summary

MedDiag Pro is a professional-grade, offline-first AI medical diagnostic desktop application powered by a Random Forest machine learning model. It identifies probable diseases from user-supplied symptoms, cross-referencing 130+ clinically sourced symptoms against 56 diseases and returning up to four ranked predictions with confidence scores, descriptions, and precautions.

Version 2.0 introduces a self-training feedback loop, a persistent star-based rating system, animated UI transitions, and a fully modular dataset architecture. These additions allow the system to adapt its predictions to real-world user preferences over time, making MedDiag Pro smarter with every session.

Disclaimer: MedDiag Pro is intended for informational and educational purposes only. It does not constitute medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional for clinical decisions.

2.  System Overview

2.1  Key Capabilities
•	AI-powered disease prediction using Random Forest (200 estimators)
•	Fuzzy symptom matching — handles typos and partial inputs automatically
•	Top 4 disease results ranked by star rating first, then model confidence
•	Persistent user feedback via star ratings saved across all sessions
•	One vote per disease per session with in-place star removal
•	Self-retraining on user preference — model improves with each vote
•	Animated card cascade UI with professional blue and black theme
•	Modular codebase — dataset separated into medical_dataset.py

2.2  Technology Stack
Component	Technology	Detail
UI Framework	CustomTkinter	Latest stable
ML Model	Scikit-learn RandomForestClassifier	200 estimators, random_state=42
Data Processing	Pandas + NumPy	Standard scientific stack
Input Matching	difflib.get_close_matches	cutoff = 0.55
Persistence	JSON (ratings.json)	Auto-created on first vote
Threading	Python threading module	Daemon background threads
Language	Python	3.8 or higher required

3.  File Structure & Architecture

The application consists of two Python source files and one auto-generated JSON file, all located in the same directory:

File	Type	Purpose
claude.py	Main application	UI, ML model, analysis engine, feedback system, navigation
medical_dataset.py	Dataset module	DISEASE_DATA, DESCRIPTION_DATA, PRECAUTION_DATA, SEVERITY_DATA
ratings.json	Auto-generated	Persistent star ratings — created automatically on first vote

Both claude.py and medical_dataset.py must reside in the same folder. The application uses a dynamic import path so it works regardless of where you place the folder.

 
4.  Installation & Setup

4.1  Prerequisites
Python	Python 3.8 or higher  —  download from python.org

pip	pip package manager (bundled with Python 3.4+)

Terminal	Command Prompt (Windows) · Terminal (macOS / Linux)

4.2  Install Dependencies
Open your terminal and run the single command below. This installs all four required libraries:

pip install customtkinter scikit-learn pandas numpy

4.3  Folder Setup
Place both source files in the same directory, for example C:\Users\You\MedDiag\ on Windows or ~/MedDiag/ on Mac/Linux:
•	claude.py  —  Main application file
•	medical_dataset.py  —  Dataset module (must be co-located)

4.4  Running the Application
# Navigate to your project folder
cd path/to/MedDiag

# Launch MedDiag Pro
python claude.py

# macOS / Linux alternative
python3 claude.py

You may also double-click claude.py if Python is associated with .py files on your operating system. The application opens as a native desktop window — no browser required.

5.  User Guide

5.1  Running a Diagnosis
1	Launch the application by running python claude.py in your terminal.
2	Type symptoms in the input bar at the top. Separate multiple symptoms with commas or spaces. Multi-word symptoms use underscores, e.g. joint_pain or high_fever.
3	Press the DIAGNOSE button or hit Enter. A progress bar shows each processing stage.
4	Up to four ranked disease cards appear in the results area below.
5	Each card shows disease name, confidence bar, star rating, description, and precautions.
6	Vote on a result, view history, or navigate to Ratings via the left sidebar.

5.2  Symptom Input Formats
The fuzzy-matching engine accepts varied input and tolerates minor typos (cutoff 0.55):

Input Style	Example
Comma-separated	fever, headache, vomiting
Space-separated	fever headache vomiting
Underscore (multi-word)	joint_pain, high_fever, breathlessness
Minor typos tolerated	heedache, vomitig, fetigue
Mixed formats	fever joint_pain, breathlessness

5.3  Understanding Each Result Card
•	Rank badge: Gold / Silver / Bronze / 4th — order reflects stars then AI confidence
•	Confidence bar: Visual percentage of the model’s predicted probability
•	Star rating row: Gold stars = accumulated user votes. Up to 4 icons; overflow shown as +N
•	About this condition: Clinical description of the disease
•	Recommended precautions: Up to 4 actionable precaution steps in a 2-column grid

 
6.  Star Rating & Feedback System

The rating system lets users express a preference for a diagnosis result. Ratings accumulate across sessions and influence both the display order and the model’s self-training. All votes are persisted in ratings.json so they survive app restarts.

6.1  Casting a Vote
•	Click the “I prefer this” button on any disease card.
•	Each disease is limited to one vote per session (closing and reopening the app resets session state).
•	After voting, the button turns grey and becomes disabled for that session.

6.2  Removing a Vote
•	After voting, hover over any gold star — it turns red as a visual cue.
•	Click any gold star to remove the vote. The count decrements, star reverts to empty, button re-enables.
•	The updated rating is saved to ratings.json immediately.

6.3  Star Display Behaviour
Vote Count	Display
0 votes	☆☆☆☆  (all empty)  —  shows “No votes yet”
1–4 votes	★...☆  (filled left to right)  —  e.g. 2 votes = ★★☆☆
5+ votes	★★★★ +N  —  overflow counter appears after the 4 icons
Voted session	All filled stars turn red on hover — click to remove

6.4  Result Ordering (Sort Priority)
1.	Star rating descending — highest-voted disease appears first in the card list.
2.	AI model confidence descending — breaks ties within the same star tier.

Even a disease with low model confidence will rank above a higher-confidence disease if it has more user votes. This is intentional — user preference overrides raw probability.

7.  Self-Training AI Engine

MedDiag Pro includes an adaptive self-training mechanism that reinforces the Random Forest classifier based on user feedback. The model becomes progressively better aligned with real-world preferences over a session.

7.1  Training Pipeline
A	User runs a diagnosis and receives up to four disease predictions.
B	User clicks “I prefer this” on a disease card.
C	The app appends 3 new training records: (preferred disease, current matched symptoms) to the in-memory training buffer.
D	The Random Forest is retrained in a daemon background thread — the UI remains fully responsive.
E	On completion, a “✦ MODEL UPDATED” flash appears in the status bar. Future diagnoses reflect the reinforced weighting.

Self-training data is held in memory only and resets when the application is closed. Star ratings and vote counts persist permanently in ratings.json across all sessions.

 
8.  Navigation & Screens

Screen	Description
🏠  Dashboard	Welcome screen with quick-fill symptom examples. The input field and DIAGNOSE button are always visible at the top.
⭐  Ratings	Leaderboard of all diseases sorted by total votes. Shows medal icons for top 3, vote counts, and a 5-star visual row for each entry.
📋  History	Chronological log of all analyses run in the current session, showing the original input and the matched symptom tokens.
ℹ️  About	System information card: AI engine details, disease and symptom counts, feature list, and technical architecture notes.

9.  Dataset Reference

All medical data lives in medical_dataset.py. This module is imported automatically by claude.py and must remain in the same directory. It exposes four data structures:

9.1  Data Structures
Variable	Python Type	Contents
DISEASE_DATA	List[Tuple[str, List[str]]]	~4,900 training records: (disease_name, [symptom list])
DESCRIPTION_DATA	Dict[str, str]	56 disease names → clinical description strings
PRECAUTION_DATA	Dict[str, List[str]]	56 disease names → 1–4 precaution step strings
SEVERITY_DATA	Dict[str, int]	130+ symptom names → severity weights (1–10)

9.2  Severity Scoring
Symptom severity weights are summed during analysis to produce an overall severity level shown in the results summary banner:

Severity Level	Score Range	UI Colour
LOW	Total < 15 points	Green  (#00E5A0)
MODERATE	15 – 29 points	Amber  (#FFAB40)
HIGH	30+ points	Red    (#FF5252)

9.3  Extending the Dataset
To add a new disease, edit medical_dataset.py only — no changes to claude.py are needed:
•	DISEASE_DATA: Append tuples of ("Disease Name", ["symptom_a", "symptom_b", ...]).
•	DESCRIPTION_DATA: Add "Disease Name": "Clinical description string."
•	PRECAUTION_DATA: Add "Disease Name": ["precaution 1", "precaution 2", ...].
•	SEVERITY_DATA: Add any new symptom names with an integer severity weight 1–10.

 
10.  Troubleshooting

Error / Symptom	Resolution
ModuleNotFoundError: customtkinter	Run:  pip install customtkinter
ModuleNotFoundError: medical_dataset	Place medical_dataset.py in the same folder as claude.py
ModuleNotFoundError: sklearn	Run:  pip install scikit-learn
No results / “No match found”	Use underscores for multi-word symptoms (joint_pain). Try common terms: fever, cough, fatigue, vomiting
python is not recognized	Python not in PATH. Reinstall Python and tick “Add to PATH” during setup
Ratings not saving	Ensure the app folder has write permissions. ratings.json is created automatically on first vote
Window too small / clipped	Resize the window — minimum supported size is 960 × 680 pixels
MODEL UPDATED not appearing	Only triggers after casting a vote. Run a diagnosis first, then click “I prefer this”
Stars not turning red on hover	You must have voted this session. Stars only become interactive after a vote is cast

11.  Technical Reference

11.1  Core Classes
Class	Responsibility
MedicalDiagnosticSystem	Main CTk window. Owns the ML model, UI layout, analysis thread, history list, and self-training logic.
StarRatingBar	CTkFrame widget rendered on each disease card. Manages vote state, star display, hover effects, and vote removal.

11.2  Key Methods
Method	File	Description
_load_and_train()	claude.py	Builds feature matrix from DISEASE_DATA + feedback_training list, fits the Random Forest.
start_analysis()	claude.py	Validates input, disables UI, spawns _run() daemon thread.
_run(raw)	claude.py	Tokenises input, fuzzy-matches symptoms, runs model.predict_proba(), calls _show_results().
_show_results()	claude.py	Renders summary banner and up to 4 disease cards with staggered animate_in() calls.
_sort_results()	claude.py	Sorts top_idx by (star rating DESC, confidence DESC) before display.
_on_feedback(disease)	claude.py	Appends 3 training records to feedback_training, spawns retrain background thread.
_retrain_background()	claude.py	Calls _load_and_train() off main thread; fires _retrain_flash() on completion via after().
animate_in(widget)	claude.py	Slide-grow animation: grows widget from height=1 to full height over N steps.
StarRatingBar._clicked()	claude.py	Records vote in SESSION_VOTES, increments ratings dict, saves JSON, refreshes stars.
StarRatingBar._remove()	claude.py	Clears SESSION_VOTES, decrements rating, saves JSON, re-enables prefer button.

11.3  Global State
Variable	Scope	Description
SESSION_VOTES	Module-level dict	Maps disease name → bool. Resets each app launch. Prevents double-voting per session.
COLORS	Module-level dict	All 20+ UI colour hex constants. Edit here to retheme the entire application.
RATINGS_FILE	Module-level str	Absolute path to ratings.json, resolved relative to the script’s own directory.
feedback_training	Instance list (app)	In-memory buffer of extra training tuples added by user votes. Cleared on app exit.
last_matched_symptoms	Instance list	Symptoms from the most recent analysis. Used as feature vector for self-training records.

 
12.  Changelog

Version 2.0
•	NEW:  Top 4 predictions displayed (up from 3), sorted by star rating then AI confidence
•	NEW:  Star rating system — one vote per session per disease, persistent via ratings.json
•	NEW:  Vote removal by clicking any gold star (hover turns red as visual cue)
•	NEW:  Overflow counter (+N) renders after 4 star icons are filled
•	NEW:  “N people rated this” natural-language vote count label on each card
•	NEW:  Self-training engine — each vote appends 3 reinforcement training records then retrains RF
•	NEW:  Ratings leaderboard navigation page with medal icons for top 3 voted diseases
•	NEW:  Modular dataset architecture — medical_dataset.py separated from main app
•	NEW:  Animated card cascade with staggered 120 ms slide-in effect per card
•	FIX:  sklearn feature name warning suppressed by passing named DataFrame to predict_proba

Version 1.0
•	Initial release — 56 diseases, 130+ symptoms, top 3 results
•	Random Forest classifier with 200 estimators
•	Fuzzy symptom matching via difflib
•	Severity scoring and precaution display grid

 

⚕
MedDiag Pro  v2.0
AI Medical Diagnostic System
For informational purposes only — not a substitute for professional medical advice.


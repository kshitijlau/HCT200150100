# app.py
import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import io

# --- Definitive Prompt ---
# The entire prompt engineering logic is stored in this multi-line string.
DEFINITIVE_PROMPT = """
# PROMPT: Generate Expert Candidate Assessment Summary

<persona>
You are an expert Assessment Analyst and professional writer for a leading leadership development firm. Your writing style is insightful, constructive, neutral, and professional, using American English. You write exclusively in the third person and present tense. You are a master at translating quantitative competency scores into a personalized, qualitative summary that is both encouraging and clear. Your primary goal is to create a seamless, flowing narrative, closely following the style and tone of the provided exemplars. You must avoid jargon, robotic phrasing, and repetitive sentence structures. You will rigorously adhere to the gender-specific pronouns provided in the candidate data.
</persona>

<context>
This report summarizes a candidate's performance on a leadership assessment. The assessment measures 8 core competencies grouped as follows. You must never refer to these competency names in your final output.

**Competency Groups:**
* **Overall Leadership Group (Determines the opening sentence and potential strengths):**
    * Overall Leadership
    * Reasoning & Problem Solving
* **Level-Specific Group (Forms the main body of the summary):**
    * Drives Results
    * Develops Talent
    * Manages Stakeholders
    * Thinks Strategically
    * Solves Challenges
    * Steers Change
</context>

<interpretation_rules>
You must follow these rules with absolute precision.

**PART A: OVERALL LEADERSHIP GROUP INTERPRETATIONS**
This text is used for the opening sentence and for potential strengths/development areas in the bullet points.

* **Competency: Overall Leadership**
    * *High (3.5-5.00):* "Demonstrates high potential for growth and success in a more complex role."
    * *Moderate-High (3.0-3.49):* "Candidate demonstrates above average potential for growth and success in a more complex role."
    * *Moderate-Low (2.5-2.99):* "Candidate demonstrates average potential for growth and success in a more complex role."
    * *Low (1.00-2.49):* "Candidate demonstrates low potential for growth and success in a more complex role."
* **Competency: Reasoning & Problem Solving**
    * *High (3.5-5.00):* "Candidate demonstrates a higher-than-average reasoning and problem-solving ability as compared to a group of peers."
    * *Moderate (2.5-3.49):* "Candidate demonstrates an average reasoning and problem-solving ability as compared to a group of peers."
    * *Low (1.00-2.49):* "Candidate demonstrates a below-average reasoning and problem-solving ability as compared to a group of peers."

**PART B: LEVEL-SPECIFIC GROUP INTERPRETATIONS**
Use the text corresponding to the candidate's assigned Level: APPLY, SHAPE, or GUIDE.

---
***LEVEL: APPLY***
---
* **Competency: Drives Results**
    * *High (3.5-5.00):* "Consistently demonstrates high motivation and initiative to exceed expectations. A strong drive to achieve goals, targets, and results. Seeks fulfillment through impact. High focus on achieving outcomes against set targets and delivers consistent performance to exceed own goals. Shows perseverance and determination to achieve tasks and goals despite challenges."
    * *Moderate (2.5-3.49):* "Demonstrates motivation and takes initiative occasionally. Demonstrates a drive to achieve goals, but may need support. Interest in making an impact is present but not sustained. Moderate focus on outcomes and performance tracking; may occasionally lack focus. Shows perseverance to achieve tasks but may require support in overcoming setbacks or challenges."
    * *Low (1.00-2.49):* "Demonstrates limited motivation or initiative; may meet expectations but does not show a consistent drive to exceed them. Fulfillment from work or desire to make an impact is not clearly evident. Low focus on outcomes; may not track performance against goals consistently. There may be a lack of perseverance and problem-solving when faced with setbacks."
* **Competency: Develops Talent**
    * *High (3.5-5.00):* "Consistently takes time to focus on both personal and professional growth - for both self and others. Actively pursues continuous improvement and excellence; shows clear willingness to learn and unlearn. Strong ability to resolve problems with team members proactively and achieve common goals. Makes contributions on a continual basis, creates trust and teamwork."
    * *Moderate (2.5-3.49):* "Focuses on personal and professional growth and engages in learning activities but may not do so consistently. Moderate openness to learning and unlearning. Cooperates with team members in most situations but may need guidance to work through conflicts. Makes contributions intermittently and may not always address conflicts when they arise."
    * *Low (1.00-2.49):* "Rarely focuses on personal or professional growth. Engagement in learning is limited and may resist feedback or change. Seldom works collaboratively with team members. Rarely contributes meaningfully and may avoid resolving conflicts, often leaving issues unaddressed."
* **Competency: Manages Stakeholders**
    * *High (3.5-5.00):* "Consistently shows capability to lead and inspire others. Displays strong empathy, understanding, and a focus on people. Builds relationships with ease and enjoys social interaction. Strong ability to identify and build relationships and connections. Understands stakeholder needs and mutual interests. Works to build long-term relationships."
    * *Moderate (2.5-3.49):* "Displays some ability to lead and inspire others. May show empathy and focus on people but not consistently. Builds relationships but may need support. May have only partial understanding of stakeholder needs and mutual interests. Works to build long-term relationships but may be inconsistent."
    * *Low (1.00-2.49):* "Demonstrates limited capability in leading or inspiring others. Social interaction may be minimal or strained. Struggles to build and maintain relationships. Demonstrates limited understanding of stakeholder needs or interdependencies, and does not work to build long-term relationships."
* **Competency: Thinks Strategically**
    * *High (3.5-5.00):* "Approaches work with a strong focus on the bigger picture. Operates independently with minimal guidance. Demonstrates a commercial and strategic mindset, regularly anticipating trends and their impact. Understands potential risks and seeks guidance to address the issues. Strong ability to revise strategies based on team needs while prioritising tasks accordingly in order to meet set deadlines."
    * *Moderate (2.5-3.49):* "Demonstrates awareness of the bigger picture but may need occasional guidance. Understands strategy in parts but may not consistently anticipate trends or broader implications. Can identify risks with some guidance and seeks input occasionally to address issues. Demonstrates some ability to revise plans but may need reminders to prioritise effectively."
    * *Low (1.00-2.49):* "Focus tends to be on immediate tasks. Requires frequent guidance. Displays limited awareness of trends or the strategic impact of work. Low ability to align goals with team direction and recognise potential risks. Requires frequent support to address issues and struggles to revise plans independently."
* **Competency: Solves Challenges**
    * *High (3.5-5.00):* "Consistently addresses problems and challenges with confidence and resilience. Takes a diligent, practical, and solution-focused approach to solving issues. Will likely remain composed in the face of setbacks and approach problems with a positive “can do” attitude."
    * *Moderate (2.5-3.49):* "Demonstrates ability to address problems but may need support or time to build confidence and resilience. Attempts a practical approach but not always solution-focused. Moderate ability to identify issues proactively, and takes action when promoted. Sometimes may struggle to remain composed under pressure."
    * *Low (1.00-2.49):* "Struggles to address problems confidently. May rely heavily on others and may not take a practical or solution-oriented approach. Does not prioritise working with others to solve problems and identify solutions. Struggles to remain composed under pressure or maintain a positive approach."
* **Competency: Steers Change**
    * *High (3.5-5.00):* "Thrives in change and complexity in the workplace. Manages new ways of working with adaptability, flexibility, and a decisiveness during uncertainty. Supports implementation of new change initiatives and takes appropriate follow-up action."
    * *Moderate (2.5-3.49):* "Generally copes with change and can adapt when needed. May need support to remain flexible or decisive in uncertain situations. Operates with a degree of comfort when facts are not fully available and support change initiatives, but follow-up action may be delayed or inconsistent."
    * *Low (1.00-2.49):* "Struggles with change or uncertainty. May resist new ways of working and has difficulty adapting or deciding in changing circumstances. May be uncomfortable operating when facts are unclear and is unlikely to support change initiatives."

---
***LEVEL: SHAPE***
---
* **Competency: Drives Results**
    * *High (3.5-5.00):* "Consistently demonstrates high motivation and initiative to exceed expectations. A strong drive to achieve goals, targets, and results. Seeks fulfillment through impact. Drives a high-performance culture across teams and demonstrates grit and persistence when working toward ambitious targets."
    * *Moderate (2.5-3.49):* "Demonstrates motivation and takes initiative occasionally. Demonstrates a drive to achieve goals, but may need support. Interest in making an impact is present but not sustained. Moderate ability to articulate performance standards that contribute to achieving organisational goals. Occasionally supports performance across teams and shows persistence when working towards goals."
    * *Low (1.00-2.49):* "Demonstrates limited motivation or initiative; may meet expectations but does not show a consistent drive to exceed them. Fulfillment from work or desire to make an impact is not clearly evident. Low ability to articulate performance standards that support organisational goals. Needs development in fostering a high-performance culture and in maintaining persistence when faced with challenging goals."
* **Competency: Develops Talent**
    * *High (3.5-5.00):* "Consistently takes time to focus on both personal and professional growth - for both self and others. Actively pursues continuous improvement and excellence; shows clear willingness to learn and unlearn. Strongly supports development of others by identifying and leveraging individual strengths. Advocates for learning and career growth, contributing to a culture of learning and continuous improvement."
    * *Moderate (2.5-3.49):* "Focuses on personal and professional growth for self and others and engages in learning activities but may not do so consistently. Displays willingness to learn and unlearn. Recognizes others’ development needs and offers support, though may not consistently nurture growth or advocate for talent advancement."
    * *Low (1.00-2.49):* "Rarely focuses on personal or professional growth- for both self and others. Engagement in learning is limited and may resist feedback or change. Shows minimal interest in developing others or contributing to a learning environment. May neglect or avoid growth conversations."
* **Competency: Manages Stakeholders**
    * *High (3.5-5.00):* "Consistently shows capability to lead and inspire others. Displays strong empathy, understanding, and a focus on people. Builds relationships with ease and enjoys social interaction. Demonstrates strong ability to engage key stakeholders, build trust-based relationships, and find synergies for mutual outcomes. Proactively networks and stays connected across internal and external touchpoints."
    * *Moderate (2.5-3.49):* "Displays some ability to lead and inspire others. May show empathy and focus on people inconsistently. Moderate ability to maintain and build relationships with key stakeholders. Often identifies synergies for positive outcomes. Occasionally proactively networks."
    * *Low (1.00-2.49):* "Demonstrates limited capability in leading or inspiring others. Social interaction may be minimal or strained. Struggles to build and maintain relationships. Rarely engages with stakeholders and does not leverage relationships for mutual outcomes. Limited presence in networks or cross-functional collaboration."
* **Competency: Thinks Strategically**
    * *High (3.5-5.00):* "Approaches work with a strong focus on the bigger picture. Operates independently with minimal guidance. Demonstrates a commercial and strategic mindset, regularly anticipating trends and their impact. Effectively balances short-term goals with long-term organizational value. Translates complex goals into clear team actions and helps others understand broader implications."
    * *Moderate (2.5-3.49):* "Demonstrates some awareness of the bigger picture but may need occasional guidance. Understands strategy in parts but may not consistently anticipate trends or broader implications. Occasionally translates organisational goals into meaningful actions. Can focus on both immediate and longer-term needs but may favor one over the other."
    * *Low (1.00-2.49):* "Focus tends to be on immediate tasks. Requires frequent guidance. Displays limited awareness of trends or the strategic impact of work. Needs ongoing guidance to connect work with strategic direction. Struggles to translate organizational priorities into meaningful tasks or influence direction."
* **Competency: Solves Challenges**
    * *High (3.5-5.00):* "Consistently addresses problems and challenges with confidence and resilience. Takes a diligent, practical, and solution-focused approach. Comfortable navigating ambiguity and complexity. Makes sound decisions under pressure and thrives in environments with multiple demands."
    * *Moderate (2.5-3.49):* "Has the ability to address problems but may need time or support to build confidence and resilience. Attempts a practical approach but not always solution-focused. Moderate ability to handle ambiguity and complex environments. Shows some confidence in leading through uncertain environments."
    * *Low (1.00-2.49):* "Struggles to address problems confidently. May rely heavily on others. Practical or solution-oriented approaches are limited. Avoids complexity and ambiguity. Rarely takes initiative in resolving obstacles."
* **Competency: Steers Change**
    * *High (3.5-5.00):* "Thrives in change and complexity. Manages new ways of working with adaptability, flexibility, and decisiveness during change. Plays an active role in transformation initiatives, shows strong resilience, and enables buy-in and alignment from others during change."
    * *Moderate (2.5-3.49):* "Demonstrates ability to cope with change and can adapt when needed. May need support to remain flexible or decisive in uncertain situations. Contributes to organisational change initiatives, may enable buy-in and shows resilience during challenging times."
    * *Low (1.00-2.49):* "Struggles with change or uncertainty. May resist new ways of working and has difficulty adapting or deciding in changing circumstances. Rarely contributes to transformation efforts and finds it difficult to stay resilient under shifting demands. Has difficulty enabling buy-in and support."

---
***LEVEL: GUIDE***
---
* **Competency: Drives Results**
    * *High (3.5-5.00):* "Consistently demonstrates high motivation and initiative to exceed expectations. A strong drive to achieve goals, targets, and results. Seeks fulfillment through impact. Supports and guides team to deliver goals on time. Recognizes high performance, addresses underperformance, displays grit, and manages resources effectively."
    * *Moderate (2.5-3.49):* "Demonstrates motivation and takes initiative occasionally. Demonstrates a drive to achieve goals, but may need support. Interest in making an impact is present but not sustained. Supports team delivery but may need prompting. Occasionally recognizes performance and addresses underperformance. Shows some grit and manages resources with support."
    * *Low (1.00-2.49):* "Demonstrates limited motivation or initiative; may meet expectations but does not show a consistent drive to exceed them. Fulfillment from work or desire to make an impact is not clearly evident. Limited support for team delivery. Rarely recognizes performance or addresses underperformance. Struggles with grit and resource management."
* **Competency: Develops Talent**
    * *High (3.5-5.00):* "Consistently takes time to focus on both personal and professional growth - for both self and others. Actively pursues continuous improvement and excellence; shows clear willingness to learn and unlearn. Coaches key talent with timely, constructive feedback. Builds capability by offering challenging development opportunities."
    * *Moderate (2.5-3.49):* "Focuses on personal and professional growth for self and others and engages in learning activities but may not do so consistently. Displays willingness to learn and unlearn. Provides feedback and guidance, though not always timely or targeted. Offers some development opportunities, but impact may vary."
    * *Low (1.00-2.49):* "Rarely focuses on personal or professional growth- for both self and others. Engagement in learning is limited and may resist feedback or change. Rarely provides meaningful feedback or development. Struggles to coach talent or build individual capability."
* **Competency: Manages Stakeholders**
    * *High (3.5-5.00):* "Consistently shows capability to lead and inspire others. Displays strong empathy, understanding, and a focus on people. Builds relationships with ease and enjoys social interaction. Builds strong relationships to achieve team goals. Understands stakeholder interests and creates long-term partnerships through relationship-building efforts."
    * *Moderate (2.5-3.49):* "Displays some ability to lead and inspire others. May show empathy and focus on people inconsistently. Moderate ability to maintain and build relationships with key stakeholders. Builds relationships when needed to meet goals. Some awareness of stakeholder interests. Maintains connections, but may not actively deepen them."
    * *Low (1.00-2.49):* "Demonstrates limited capability in leading or inspiring others. Social interaction may be minimal or strained. Struggles to build and maintain relationships. Engages with stakeholders minimally. Limited understanding of mutual interests. Rarely invests in building or maintaining long-term relationships."
* **Competency: Thinks Strategically**
    * *High (3.5-5.00):* "Approaches work with a strong focus on the bigger picture. Operates independently with minimal guidance. Demonstrates a commercial and strategic mindset, regularly anticipating trends and their impact. Considers both short- and long-term impact of decisions. Translates departmental strategy into clear, meaningful actions for self and others."
    * *Moderate (2.5-3.49):* "Demonstrates some awareness of the bigger picture but may need occasional guidance. Understands strategy in parts but may not consistently anticipate trends or broader implications. Acknowledges short- and long-term implications, though not always fully. Can link strategy to actions but may need support or clarification."
    * *Low (1.00-2.49):* "Focus tends to be on immediate tasks. Requires frequent guidance. Displays limited awareness of trends or the strategic impact of work. Focuses mostly on immediate tasks. Limited awareness of broader implications or difficulty turning strategy into clear actions."
* **Competency: Solves Challenges**
    * *High (3.5-5.00):* "Consistently addresses problems and challenges with confidence and resilience. Takes a diligent, practical, and solution-focused approach. Manages conflicting departmental and people priorities effectively and consistently weighs them when making decisions."
    * *Moderate (2.5-3.49):* "Has the ability to address problems but may need time or support to build confidence and resilience. Attempts a practical approach but not always solution-focused. Manages departmental and people priorities but may not always weigh them evenly when making decisions."
    * *Low (1.00-2.49):* "Struggles to address problems confidently. May rely heavily on others. Practical or solution-oriented approaches are limited. Struggles to manage conflicting priorities and rarely weighs them appropriately when making decisions."
* **Competency: Steers Change**
    * *High (3.5-5.00):* "Thrives in change and complexity. Manages new ways of working with adaptability, flexibility, and decisiveness during change. Acts as a role model for positive change, inspiring others and clearly translating the change journey into defined actions."
    * *Moderate (2.5-3.49):* "Demonstrates ability to cope with change and can adapt when needed. May need support to remain flexible or decisive in uncertain situations. Supports change efforts and sometimes inspires others, but may need help translating the journey into clear actions."
    * *Low (1.00-2.49):* "Struggles with change or uncertainty. May resist new ways of working and has difficulty adapting or deciding in changing circumstances. Rarely acts as a role model for change and struggles to inspire or define clear actions in the change journey."

**PART C: SUMMARY STRUCTURE AND EXECUTION**
1.  **Opening Sentence:** Your summary MUST begin with the exact sentence from the 'Overall Leadership' interpretation text (Part A) that corresponds to the candidate's score. Add the candidate's first name to the beginning.
2.  **Main Body Paragraph:** Following the opening, describe the 6 'Level-Specific Group' competencies by internally sorting their scores in descending order and weaving their corresponding interpretation texts from Part B into a natural paragraph, mirroring the style of the exemplars. DO NOT name the competencies.
3.  **Bullet Points:** Provide two strengths and two development areas based on the highest and lowest scores. The bullet points MUST NOT repeat sentences from the main summary. They must be complementary, behavioral statements derived from the interpretation text, as shown in the exemplars.

</interpretation_rules>

<exemplars>
Here are four golden standard examples. Study them carefully to understand the expected narrative style, tone, and structure, and how to adapt the summary for different lengths.

**--- EXAMPLE 1 ---**
<candidate_data>
* Name: Sub 1
* Gender (for pronouns): She/Her
* Level: Apply
* Scores:
    * Overall Leadership: 4
    * Reasoning & Problem Solving: 4
    * Drives Results: 4
    * Develops Talent: 3
    * Manages Stakeholder: 4
    * Thinks Strategically: 4
    * Solves Challenges: 5
    * Steers Change: 4
</candidate_data>
<sme_written_output>
{
  "summary_200": "Sub 1 demonstrates high potential for growth and success in a more complex role. She demonstrates high motivation to exceed expectations and strong drive to achieve goals. She shows a strong capability to lead and inspire others, building relationships with ease and enjoying social interactions. She has strong focus on the bigger picture, thrives in change and complexity, and consistently addresses problems with confidence and resilience. While she focuses on personal and professional growth and engages in learning activities, she may not do so consistently and may need guidance to work through conflicts.\\n\\n**Strengths:**\\n* Takes a diligent, practical, and solution-focused approach to solving issues.\\n* Will likely remain composed in the face of setbacks and approach problems with a positive attitude.\\n\\n**Development Areas:**\\n* Could benefit from developing a greater openness to learning and unlearning.\\n* May benefit from ensuring conflicts are addressed when they arise.",
  "summary_150": "Sub 1 demonstrates high potential for growth and success. With high motivation and a strong drive for results, she capably leads and inspires others while building relationships with ease. She maintains a strong focus on the bigger picture, thrives in complexity, and addresses challenges with confidence. Her main area for development is to apply that same consistency to her personal growth and in proactively resolving team conflicts.\\n\\n**Strengths:**\\n* Takes a diligent, practical, and solution-focused approach to solving issues.\\n* Remains composed in the face of setbacks and approaches problems with a positive attitude.\\n\\n**Development Areas:**\\n* Could benefit from developing a greater openness to learning and unlearning.\\n* May benefit from ensuring conflicts are addressed when they arise."
}
</sme_written_output>

**--- EXAMPLE 2 ---**
<candidate_data>
* Name: John Doe
* Gender (for pronouns): He/Him
* Level: Apply
* Scores:
    * Overall Leadership: 3
    * Reasoning & Problem Solving: 3
    * Drives Results: 2
    * Develops Talent: 2
    * Manages Stakeholder: 3
    * Thinks Strategically: 3
    * Solves Challenges: 3
    * Steers Change: 3
</candidate_data>
<sme_written_output>
{
  "summary_200": "John Doe demonstrates moderate potential for growth and success in a more complex role. He may not show a consistent drive to exceed expectations. Engagement in learning is limited and he may resist feedback or change. He displays some ability to build relationships, and address problems, though he may need support or time to build confidence. While he demonstrates awareness of the bigger picture, he may need occasional guidance. He generally copes with change and can adapt when needed.\\n\\n**Strengths:**\\n* Works to build long-term relationships and shows some understanding of stakeholder needs.\\n* Operates with a degree of comfort when facts are not fully available and supports change initiatives.\\n\\n**Development Areas:**\\n* Could benefit from developing greater perseverance when faced with setbacks and a stronger focus on tracking outcomes against goals.\\n* Could focus on working more collaboratively with team members and proactively contributing to resolving conflicts.",
  "summary_150": "John Doe demonstrates moderate potential for growth and success. While he can adapt to change and shows an awareness of the bigger picture, he may require guidance. He displays some ability to build relationships and address problems but would benefit from building more confidence. His key development areas are increasing his consistent drive to exceed expectations and being more proactive in his engagement with learning and feedback.\\n\\n**Strengths:**\\n* Works to build long-term relationships with some understanding of stakeholder needs.\\n* Operates with a degree of comfort when facts are not fully available.\\n\\n**Development Areas:**\\n* Could benefit from greater perseverance and a stronger focus on tracking outcomes against goals.\\n* Could focus on working more collaboratively and proactively resolving conflicts."
}
</sme_written_output>

**--- EXAMPLE 3 ---**
<candidate_data>
* Name: Ayesha Obaid Al Mheiri
* Gender (for pronouns): She/Her
* Level: Shape
* Scores:
    * Overall Leadership: 2.97
    * Reasoning & Problem Solving: 3
    * Drives Results: 2.97
    * Develops Talent: 3.15
    * Manages Stakeholder: 2.92
    * Thinks Strategically: 3.38
    * Solves Challenges: 3.92
    * Steers Change: 2.89
</candidate_data>
<sme_written_output>
{
  "summary_200": "Ayesha Obaid Al Mheiri demonstrates moderate potential for growth and success in a more complex role. She occasionally takes initiatives and demonstrates motivation. She focuses on personal and professional growth but may not do so consistently. Ayesha displays the ability to lead and inspire others and build and maintain relationships. She shows awareness of the bigger picture but may need occasional guidance to translate broader goals into action. She addresses problems with confidence, applies a practical and solution-focused approach, and handles ambiguity well. She generally adapts to change when needed, though may require support to remain flexible or decisive in uncertain situations.\\n\\n**Strengths:**\\n* Consistently addresses problems and challenges with confidence and resilience.\\n* Occasionally translates organisational goals into meaningful actions.\\n\\n**Development Areas:**\\n* May need support to remain flexible or decisive in uncertain situations.\\n* May show empathy and focus on people inconsistently.",
  "summary_150": "Ayesha Obaid Al Mheiri demonstrates moderate potential for growth. She shows awareness of the bigger picture and confidently addresses problems with a solution-focused approach. She can lead and adapt to change, though may need support to remain decisive in uncertain situations. Her development would be enhanced by a more consistent focus on her personal growth and in demonstrating empathy when managing stakeholder relationships.\\n\\n**Strengths:**\\n* Consistently addresses problems and challenges with confidence and resilience.\\n* Occasionally translates organizational goals into meaningful actions.\\n\\n**Development Areas:**\\n* May need support to remain flexible or decisive in uncertain situations.\\n* May show empathy and focus on people inconsistently."
}
</sme_written_output>

**--- EXAMPLE 4 ---**
<candidate_data>
* Name: Ali Salem Al Suwaidi
* Gender (for pronouns): He/Him
* Level: Apply
* Scores:
    * Overall Leadership: 2.55
    * Reasoning & Problem Solving: 3
    * Drives Results: 2.22
    * Develops Talent: 2.55
    * Manages Stakeholder: 2.36
    * Thinks Strategically: 2.47
    * Solves Challenges: 5
    * Steers Change: 1.43
</candidate_data>
<sme_written_output>
{
  "summary_200": "Ali Salem Al Suwaidi demonstrates moderate potential for growth and success in a more complex role. He may meet expectations but has limited drive to exceed them. Fulfillment from work or a desire to make an impact is limited. He focuses on personal and professional growth but may not do so consistently. He demonstrates limited capability to lead and inspire others and struggles to build relationships. His focus tends to be on immediate tasks, he struggles to address problems, may not take a solution-oriented approach and may rely heavily on others. However, he generally copes with change and can adapt when needed.\\n\\n**Strengths:**\\n* Supports change initiatives and operate with comfort during uncertainty.\\n* Focuses on personal and professional growth and engages in learning activities, though this may be inconsistent.\\n\\n**Development Areas:**\\n* Enhance independent problem-solving and decision-making confidence.\\n* Increase motivation, initiative, and perseverance in setbacks.",
  "summary_150": "Ali Salem Al Suwaidi demonstrates moderate potential for growth. While he can cope with change when needed, his focus tends to remain on immediate tasks. He may meet expectations but shows a limited drive to exceed them and struggles to build relationships. He would benefit from developing more confidence and taking a more solution-oriented approach when addressing problems. His focus on personal growth is a good foundation to build upon.\\n\\n**Strengths:**\\n* Supports change initiatives and can operate with comfort during uncertainty.\\n* Focuses on personal and professional growth, though this may be inconsistent.\\n\\n**Development Areas:**\\n* Enhance independent problem-solving and decision-making confidence.\\n* Increase motivation, initiative, and perseverance in setbacks."
}
</sme_written_output>
</exemplars>

<task>
Analyze the following candidate data. Based on all the rules, context, and exemplars provided, generate a personalized assessment summary.

**Candidate Data:**
* **Name:** [Candidate Name]
* **Gender (for pronouns):** [He/Him, She/Her, They/Them]
* **Level:** [Apply/Shape/Guide]
* **Scores:**
    * Overall Leadership: [Score]
    * Reasoning & Problem Solving: [Score]
    * Drives Results: [Score]
    * Develops Talent: [Score]
    * Manages Stakeholders: [Score]
    * Thinks Strategically: [Score]
    * Solves Challenges: [Score]
    * Steers Change: [Score]

Your final output must be a single, raw JSON object with three keys: "summary_200", "summary_150", and "summary_100". The value for each key will be the complete summary (paragraph and bullet points) at that approximate word count. Do not include any other text, explanation, or markdown formatting like ```json outside of this JSON object.
</task>
"""

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Assessment Summary Generator",
    page_icon="✍️",
    layout="wide"
)

# --- App State Management ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# --- Helper Functions ---

def create_sample_excel():
    """Creates an in-memory sample Excel file for users to download."""
    sample_data = {
        'Name': ['Jane Doe', 'John Smith'],
        'Gender': ['She/Her', 'He/Him'],
        'Level': ['Apply', 'Shape'],
        'Overall Leadership': [4.1, 2.8],
        'Reasoning & Problem Solving': [3.5, 3.1],
        'Drives Results': [4.5, 2.5],
        'Develops Talent': [3.9, 3.2],
        'Manages Stakeholders': [4.2, 2.9],
        'Thinks Strategically': [3.8, 3.4],
        'Solves Challenges': [4.0, 3.8],
        'Steers Change': [3.7, 2.7]
    }
    df = pd.DataFrame(sample_data)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Candidates')
    processed_data = output.getvalue()
    return processed_data

def generate_summaries_for_candidate(row, model):
    """Constructs the prompt and calls the Gemini API for a single candidate."""
    try:
        task_prompt = f"""
<task>
Analyze the following candidate data. Based on all the rules, context, and exemplars provided, generate a personalized assessment summary.

**Candidate Data:**
* **Name:** {row['Name']}
* **Gender (for pronouns):** {row['Gender']}
* **Level:** {row['Level']}
* **Scores:**
    * Overall Leadership: {row['Overall Leadership']}
    * Reasoning & Problem Solving: {row['Reasoning & Problem Solving']}
    * Drives Results: {row['Drives Results']}
    * Develops Talent: {row['Develops Talent']}
    * Manages Stakeholders: {row['Manages Stakeholders']}
    * Thinks Strategically: {row['Thinks Strategically']}
    * Solves Challenges: {row['Solves Challenges']}
    * Steers Change: {row['Steers Change']}

Your final output must be a single, raw JSON object with three keys: "summary_200", "summary_150", and "summary_100". The value for each key will be the complete summary (paragraph and bullet points) at that approximate word count. Do not include any other text, explanation, or markdown formatting like ```json outside of this JSON object.
</task>
"""
        
        final_prompt = DEFINITIVE_PROMPT.replace(
            """<task>
Analyze the following candidate data. Based on all the rules, context, and exemplars provided, generate a personalized assessment summary.

**Candidate Data:**
* **Name:** [Candidate Name]
* **Gender (for pronouns):** [He/Him, She/Her, They/Them]
* **Level:** [Apply/Shape/Guide]
* **Scores:**
    * Overall Leadership: [Score]
    * Reasoning & Problem Solving: [Score]
    * Drives Results: [Score]
    * Develops Talent: [Score]
    * Manages Stakeholders: [Score]
    * Thinks Strategically: [Score]
    * Solves Challenges: [Score]
    * Steers Change: [Score]

Your final output must be a single, raw JSON object with three keys: "summary_200", "summary_150", and "summary_100". The value for each key will be the complete summary (paragraph and bullet points) at that approximate word count. Do not include any other text, explanation, or markdown formatting like ```json outside of this JSON object.
</task>""",
            task_prompt
        )

        response = model.generate_content(final_prompt)
        
        cleaned_response = response.text.strip().lstrip("```json").rstrip("```").strip()
        
        summaries = json.loads(cleaned_response)
        return summaries.get('summary_200', 'Error'), summaries.get('summary_150', 'Error'), summaries.get('summary_100', 'Error')

    except Exception as e:
        error_message = f"Failed to process {row.get('Name', 'Unknown Candidate')}. Error: {str(e)}. Raw Response: {response.text if 'response' in locals() else 'N/A'}"
        st.warning(error_message)
        return "Error", "Error", "Error"

# --- Main App UI ---

st.title("✍️ AI Assessment Summary Generator")
st.markdown("This application uses **Gemini 2.5 Pro** to transform quantitative assessment scores into professional, narrative summaries. Upload your candidate data below to begin.")

# --- API Key Configuration ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    st.sidebar.success("API Key loaded successfully!", icon="✅")
except Exception as e:
    st.error("🚨 Google API Key not found or invalid in secrets.toml. Please ensure it is set up correctly for deployment.")
    st.stop()

# --- Sidebar for Instructions and File Download ---
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1.  **Prepare Your Data**: Use the template to ensure your Excel file has the correct columns.
    2.  **Upload**: Drag and drop or browse for your Excel file.
    3.  **Generate**: Click the 'Generate All Summaries' button.
    4.  **Download**: Once processing is complete, your results will appear with a download button.
    """)
    
    st.header("Download Template")
    sample_excel_file = create_sample_excel()
    st.download_button(
        label="📥 Download Sample Excel File",
        data=sample_excel_file,
        file_name="candidate_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- File Uploader and Processing Logic ---
uploaded_file = st.file_uploader("📂 Upload Your Candidate Data Excel File", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.info(f"File '{uploaded_file.name}' uploaded successfully. Found {len(df)} candidates.")
        
        with st.expander("View Uploaded Data"):
            st.dataframe(df)

        if st.button("🚀 Generate All Summaries", type="primary"):
            st.session_state.processed_data = None
            progress_bar = st.progress(0, text="Initializing...")
            
            results_list = []
            total_rows = len(df)

            for index, row in df.iterrows():
                progress_text = f"Processing candidate {index + 1}/{total_rows}: {row['Name']}..."
                progress_bar.progress((index + 1) / total_rows, text=progress_text)
                
                summaries = generate_summaries_for_candidate(row, model)
                results_list.append(summaries)
            
            progress_bar.empty()
            
            results_df = pd.DataFrame(results_list, columns=['Summary (200 words)', 'Summary (150 words)', 'Summary (100 words)'])
            
            final_df = pd.concat([df, results_df], axis=1)
            st.session_state.processed_data = final_df
            
            st.balloons()
            st.success("🎉 All summaries generated successfully!")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

# --- Display Results and Download Button ---
if st.session_state.processed_data is not None:
    st.header("Generated Summaries")
    st.dataframe(st.session_state.processed_data)
    
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        st.session_state.processed_data.to_excel(writer, index=False, sheet_name='Generated_Summaries')
    
    st.download_button(
        label="✅ Download Results as Excel",
        data=output_excel.getvalue(),
        file_name='candidate_summaries_output.xlsx',
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

from pydantic import BaseModel, Field
from typing import Optional

class ReflectionAnalysisRequest(BaseModel):
    """Request model for reflection analysis"""
    what_learned: str = Field(
        ..., 
        example="I learned the basics of Go programming, including variables, functions, and goroutines",
        description="What the user learned (text)"
    )
    how_felt: str = Field(
        ..., 
        example="I felt excited and confident, though some concepts were challenging at first",
        description="How the user felt after learning (emotional reflection text)"
    )
    learning_progress_score: int = Field(
        ..., 
        ge=1, 
        le=5, 
        example=4,
        description="User's self-assessed learning progress score (1–5)"
    )
    challenge_level_score: int = Field(
        ..., 
        ge=1, 
        le=5, 
        example=4,
        description="Challenge level of the learning (1–5)"
    )


class ReflectionAnalysisResponse(BaseModel):
    """Response model for reflection analysis"""
    reflection_score: int = Field(
        ..., 
        ge=1, 
        le=10,
        description="Overall reflection score (1–10)"
    )
    scoring_conclusion: str = Field(
        ..., 
        description="A single sentence explaining why the score was assigned"
    )
    sentiment_classification: str = Field(
        ...,
        description="Sentiment of the 'how_felt' text (positive, neutral, or negative)"
    )
    used_context: list[str] = Field(
        default=[],
        description="RAG context used for evaluation (Debugging)"
    )


# ==================== EXAMPLE REFLECTIONS FOR REFERENCE ====================
# These examples are organized by sentiment category to help understand different
# reflection patterns and how they should be evaluated.

# POSITIVE SENTIMENT EXAMPLES
POSITIVE_REFLECTIONS = [
    {
        "what_learned": "The lesson revealed how REST APIs manage communication between different systems in a structured way.",
        "how_felt": "A strong sense of confidence came from finally seeing the full picture."
    },
    {
        "what_learned": "Practical exercises clarified the role of SQL joins in connecting related data across tables.",
        "how_felt": "Satisfaction followed once the queries started producing meaningful results."
    },
    {
        "what_learned": "Step-by-step examples made FastAPI routing and validation feel more intuitive than expected.",
        "how_felt": "Motivation increased as backend development became less intimidating."
    },
    {
        "what_learned": "Observing how JWT tokens flow through the authentication process helped clarify API security.",
        "how_felt": "Pride emerged after understanding a concept that once felt unreachable."
    },
    {
        "what_learned": "Error messages began to feel more like guidance rather than obstacles during debugging.",
        "how_felt": "Relief and confidence replaced earlier anxiety."
    },
    {
        "what_learned": "Database design principles such as normalization became clearer through real scenarios.",
        "how_felt": "A feeling of clarity and control developed over data modeling decisions."
    },
    {
        "what_learned": "Exposure to asynchronous programming highlighted why modern applications rely on it.",
        "how_felt": "Curiosity and excitement grew while exploring its possibilities."
    },
    {
        "what_learned": "Working through deployment steps showed how applications transition from local to cloud environments.",
        "how_felt": "A sense of accomplishment came from seeing the app run successfully online."
    },
    {
        "what_learned": "The structure of MVC architecture became understandable through visual diagrams and examples.",
        "how_felt": "Confidence grew as system organization started to make sense."
    },
    {
        "what_learned": "Version control practices demonstrated how teams collaborate without breaking each other's work.",
        "how_felt": "Comfort replaced uncertainty when thinking about group projects."
    },
    {
        "what_learned": "The lesson connected frontend actions to backend logic in a clear workflow.",
        "how_felt": "Understanding this flow created a feeling of confidence."
    },
    {
        "what_learned": "Reading documentation alongside practice highlighted how developers solve problems independently.",
        "how_felt": "Empowerment followed from being less dependent on tutorials."
    },
    {
        "what_learned": "Performance optimization concepts became clearer after comparing slow and fast queries.",
        "how_felt": "Satisfaction came from noticing measurable improvements."
    },
    {
        "what_learned": "Security concepts felt less abstract once real attack scenarios were discussed.",
        "how_felt": "Confidence grew in handling basic protection mechanisms."
    },
    {
        "what_learned": "Breaking complex problems into smaller parts proved to be an effective strategy.",
        "how_felt": "A sense of control replaced previous overwhelm."
    },
    {
        "what_learned": "Testing APIs with real requests clarified how endpoints should behave.",
        "how_felt": "Assurance developed from being able to verify correctness."
    },
    {
        "what_learned": "The relationship between data models and business logic became more obvious.",
        "how_felt": "Clarity brought a calm and confident feeling."
    },
    {
        "what_learned": "Consistent coding standards demonstrated how readability improves long-term maintenance.",
        "how_felt": "Pride emerged from writing cleaner code."
    },
    {
        "what_learned": "The lesson highlighted common mistakes developers make and how to avoid them.",
        "how_felt": "Confidence increased from knowing what to watch out for."
    },
    {
        "what_learned": "Connecting multiple concepts together created a coherent understanding of the system.",
        "how_felt": "A strong sense of progress and confidence followed."
    },
]

# NEUTRAL SENTIMENT EXAMPLES
NEUTRAL_REFLECTIONS = [
    {
        "what_learned": "The overall idea of the system is clearer, but many implementation details remain fuzzy.",
        "how_felt": "Feelings remain neutral with cautious optimism."
    },
    {
        "what_learned": "Some concepts made sense immediately, while others require repeated review.",
        "how_felt": "Uncertainty still exists about full mastery."
    },
    {
        "what_learned": "The lesson provided a general framework without deep practical application.",
        "how_felt": "Feelings remain balanced without strong confidence or frustration."
    },
    {
        "what_learned": "Tutorials helped guide the process, though independent application is still difficult.",
        "how_felt": "A neutral feeling reflects partial understanding."
    },
    {
        "what_learned": "The main workflow is understandable, but edge cases are confusing.",
        "how_felt": "A mix of curiosity and hesitation remains."
    },
    {
        "what_learned": "Technical terms were introduced faster than they could be absorbed.",
        "how_felt": "Feelings remain neutral with awareness of learning gaps."
    },
    {
        "what_learned": "Understanding improved compared to before, but consistency is still lacking.",
        "how_felt": "Moderate confidence mixed with uncertainty remains."
    },
    {
        "what_learned": "The concepts are familiar, but execution still requires guidance.",
        "how_felt": "Feelings are stable but not confident."
    },
    {
        "what_learned": "Progress was made, though it feels slower than expected.",
        "how_felt": "Neutral emotions accompany steady learning."
    },
    {
        "what_learned": "Some parts of the lesson felt clear, while others were difficult to connect.",
        "how_felt": "Feelings remain mixed and neutral."
    },
    {
        "what_learned": "Exposure to new ideas helped broaden understanding, though depth is missing.",
        "how_felt": "A neutral sense of progress is present."
    },
    {
        "what_learned": "The lesson answered some questions but raised new ones.",
        "how_felt": "Curiosity exists without full confidence."
    },
    {
        "what_learned": "Basic understanding exists, but practical confidence has not developed yet.",
        "how_felt": "Feelings remain neutral and patient."
    },
    {
        "what_learned": "The topic feels more familiar, though not fully comfortable.",
        "how_felt": "Mild confidence balanced with hesitation."
    },
    {
        "what_learned": "Learning occurred, but the ability to explain it clearly is still limited.",
        "how_felt": "Feelings remain neutral and reflective."
    },
]

# NEGATIVE SENTIMENT EXAMPLES
NEGATIVE_REFLECTIONS = [
    {
        "what_learned": "The topic felt scattered, making it difficult to identify the main ideas.",
        "how_felt": "Confusion and frustration dominated the experience."
    },
    {
        "what_learned": "Explanations moved too quickly to absorb the key concepts.",
        "how_felt": "Stress and discouragement followed."
    },
    {
        "what_learned": "Despite effort, connections between concepts remained unclear.",
        "how_felt": "Feelings of frustration and self-doubt emerged."
    },
    {
        "what_learned": "Practical exercises were hard to follow without stronger fundamentals.",
        "how_felt": "Overwhelm replaced motivation."
    },
    {
        "what_learned": "Important details were missed, leading to gaps in understanding.",
        "how_felt": "A sense of being lost persisted."
    },
    {
        "what_learned": "The lesson introduced complexity without enough supporting examples.",
        "how_felt": "Frustration and confusion remained throughout."
    },
    {
        "what_learned": "Attempts to apply the knowledge resulted in repeated errors.",
        "how_felt": "Disappointment and mental fatigue followed."
    },
    {
        "what_learned": "The overall message of the lesson failed to come together clearly.",
        "how_felt": "Discouragement replaced interest."
    },
    {
        "what_learned": "Instructions were difficult to interpret, making progress slow.",
        "how_felt": "Stress and irritation increased."
    },
    {
        "what_learned": "The topic felt beyond the current level of understanding.",
        "how_felt": "Feelings of inadequacy and frustration appeared."
    },
    {
        "what_learned": "Concentration was difficult due to the complexity of the content.",
        "how_felt": "Confusion dominated the learning experience."
    },
    {
        "what_learned": "Key ideas were introduced without sufficient explanation.",
        "how_felt": "Frustration grew as understanding declined."
    },
    {
        "what_learned": "Repeated review did not significantly improve comprehension.",
        "how_felt": "Discouragement and exhaustion followed."
    },
    {
        "what_learned": "Learning outcomes felt unclear even after completing the lesson.",
        "how_felt": "Disappointment and confusion remained."
    },
    {
        "what_learned": "The session ended without a clear takeaway.",
        "how_felt": "A strong sense of frustration and demotivation persisted."
    },
]
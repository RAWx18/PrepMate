import json
import pandas as pd
import numpy as np
import typing as t
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from dataclasses import dataclass, field

@dataclass
class StudentPerformanceConfig:
    """Configuration for student performance analysis."""
    performance_threshold_low: float = 50.0
    performance_threshold_high: float = 80.0
    confidence_interval: float = 0.95
    weak_topic_count: int = 2

class AdvancedNEETStudentAnalytics:
    def __init__(self, 
                 quiz_endpoints_path: str, 
                 quiz_submission_path: str, 
                 historical_quiz_path: str,
                 config: StudentPerformanceConfig = StudentPerformanceConfig()):
        """
        Initialize analytics with data sources and configuration.
        
        Args:
            quiz_endpoints_path: Path to quiz endpoints JSON
            quiz_submission_path: Path to quiz submission data JSON
            historical_quiz_path: Path to historical quiz data JSON
            config: Performance analysis configuration
        """
        self.config = config
        self.load_data(quiz_endpoints_path, quiz_submission_path, historical_quiz_path)
        self.preprocess_data()
    
    def load_data(self, *paths):
        """Load JSON data from specified paths."""
        try:
            with open(paths[0], 'r') as f:
                self.quiz_endpoints = json.load(f)
            with open(paths[1], 'r') as f:
                self.quiz_submission = json.load(f)
            with open(paths[2], 'r') as f:
                self.historical_quizzes = json.load(f)
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self):
        """Advanced data preprocessing with enhanced feature extraction."""
        questions_df = pd.DataFrame([
            {
                'question_id': q['id'],
                'topic': q['topic'],
                'difficulty': q.get('difficulty_level', 'Unknown'),
                'description': q['description'],
                'question_type': 'multiple_choice' if len(q.get('options', [])) > 1 else 'other'
            } for q in self.quiz_endpoints['quiz']['questions']
        ])
        self.questions_df = questions_df
    
    def analyze_performance(self, user_id: str) -> dict:
        """
        Advanced performance analysis with more granular insights.
        
        Args:
            user_id: Unique identifier for the student
        
        Returns:
            Comprehensive performance metrics dictionary
        """
        user_quizzes = [quiz for quiz in self.historical_quizzes if quiz['user_id'] == user_id]
        
        if not user_quizzes:
            return {
                'total_quizzes': 0,
                'overall_performance': {
                    'average_score': 0,
                    'average_accuracy': 0,
                    'performance_variance': 0
                },
                'topic_performance': {},
                'performance_trend': {
                    'improvement_rate': 0,
                    'consistency_score': 0
                }
            }
        
        performance_metrics = {
            'total_quizzes': len(user_quizzes),
            'overall_performance': {
                'average_score': np.mean([quiz['score'] for quiz in user_quizzes]),
                'average_accuracy': np.mean([float(quiz['accuracy'].strip('%')) for quiz in user_quizzes]),
                'performance_variance': np.var([float(quiz['accuracy'].strip('%')) for quiz in user_quizzes])
            },
            'topic_performance': self._analyze_topic_performance(user_quizzes),
            'performance_trend': self._calculate_performance_trend(user_quizzes)
        }
        
        return performance_metrics
    
    def _analyze_topic_performance(self, user_quizzes):
        """Detailed topic-level performance analysis."""
        topic_performance = {}
        for quiz in user_quizzes:
            topic = quiz['quiz']['topic']
            accuracy = float(quiz['accuracy'].strip('%'))
            
            if topic not in topic_performance:
                topic_performance[topic] = {
                    'attempts': 1,
                    'total_accuracy': accuracy,
                    'max_accuracy': accuracy,
                    'min_accuracy': accuracy
                }
            else:
                tp = topic_performance[topic]
                tp['attempts'] += 1
                tp['total_accuracy'] += accuracy
                tp['max_accuracy'] = max(tp['max_accuracy'], accuracy)
                tp['min_accuracy'] = min(tp['min_accuracy'], accuracy)
        
        # Calculate average accuracy for each topic
        for topic, data in topic_performance.items():
            data['average_accuracy'] = data['total_accuracy'] / data['attempts']
        
        return topic_performance
    
    def _calculate_performance_trend(self, user_quizzes):
        """Calculate performance trend and progression."""
        sorted_quizzes = sorted(user_quizzes, key=lambda x: x['submitted_at'])
        accuracies = [float(quiz['accuracy'].strip('%')) for quiz in sorted_quizzes]
        
        trend = {
            'improvement_rate': np.polyfit(range(len(accuracies)), accuracies, 1)[0],
            'consistency_score': np.std(accuracies)
        }
        return trend
    
    def generate_student_persona(self, performance_metrics: dict) -> dict:
        """
        Advanced student persona generation with multi-dimensional insights.
        
        Args:
            performance_metrics: Comprehensive performance metrics
        
        Returns:
            Detailed student persona dictionary
        """
        # Safely extract metrics with default values
        avg_accuracy = performance_metrics.get('overall_performance', {}).get('average_accuracy', 0)
        improvement_rate = performance_metrics.get('performance_trend', {}).get('improvement_rate', 0)
        consistency_score = performance_metrics.get('performance_trend', {}).get('consistency_score', float('inf'))
        
        personas = {
            'Strategic Learner': {
                'conditions': [
                    avg_accuracy > 80,
                    improvement_rate > 0,
                    consistency_score < 10
                ],
                'description': "Methodical learner with consistent performance and upward trajectory"
            },
            'Resilient Improver': {
                'conditions': [
                    avg_accuracy > 65,
                    improvement_rate > 1,
                    consistency_score > 10
                ],
                'description': "Shows significant improvement potential with adaptable learning approach"
            }
        }
        
        for persona_name, persona_details in personas.items():
            if all(persona_details['conditions']):
                return {
                    'name': persona_name,
                    'description': persona_details['description']
                }
        
        return {'name': 'Emerging Potential', 'description': 'Developing learning strategies'}
    
    def recommend_improvement_steps(self, performance_metrics: dict) -> dict:
        """
        Comprehensive recommendation generation with targeted strategies.
        
        Args:
            performance_metrics: Detailed performance metrics
        
        Returns:
            Actionable improvement recommendations
        """
        # Handle case of no topic performance data
        topic_performance = performance_metrics.get('topic_performance', {})
        
        if not topic_performance:
            return {
                'weak_topics': [],
                'suggested_practice': [],
                'learning_strategy': "Basic Foundational Learning",
                'personalized_tips': ["Start with building core conceptual understanding"]
            }
        
        weak_topics = sorted(
            topic_performance.items(), 
            key=lambda x: x[1].get('average_accuracy', 0)
        )[:self.config.weak_topic_count]
        
        recommendations = {
            'weak_topics': [topic for topic, _ in weak_topics],
            'suggested_practice': [
                f"Focus on {topic} via advanced mock tests with detailed solution analysis" 
                for topic in [topic for topic, _ in weak_topics]
            ],
            'learning_strategy': self._select_learning_strategy(performance_metrics),
            'personalized_tips': self._generate_personalized_tips(performance_metrics)
        }
        
        return recommendations
    
    def _select_learning_strategy(self, performance_metrics):
        """Dynamically select learning strategy based on performance."""
        avg_accuracy = performance_metrics['overall_performance']['average_accuracy']
        
        if avg_accuracy < self.config.performance_threshold_low:
            return "Foundational Reconstruction: Core concept rebuilding"
        elif avg_accuracy < self.config.performance_threshold_high:
            return "Strategic Topic Reinforcement: Targeted improvements"
        else:
            return "Advanced Performance Optimization: Precision and speed enhancement"
    
    def _generate_personalized_tips(self, performance_metrics):
        """Generate context-specific learning tips."""
        tips = []
        if performance_metrics['performance_trend']['consistency_score'] > 15:
            tips.append("Focus on maintaining consistent performance across topics")
        
        if performance_metrics['performance_trend']['improvement_rate'] < 0:
            tips.append("Identify and address learning barriers causing performance decline")
        
        return tips

    def predict_neet_rank(self, user_performance_history: list) -> dict:
        """
        Predict NEET rank using available historical quiz performance data.
        
        Args:
            user_performance_history: List of historical quiz performances
        
        Returns:
            Rank prediction with confidence metrics
        """
        if len(user_performance_history) < 2:
            return {
                'predicted_rank': None,
                'confidence': 0.0,
                'warning': 'Insufficient historical data',
                'rank_range': {
                    'lower_bound': None,
                    'upper_bound': None
                }
            }
        
        # Extract meaningful features from available data
        features = []
        for performance in user_performance_history:
            features.append([
                performance['score'],
                float(performance['accuracy'].strip('%')),
                performance['total_questions'],
                len(performance['response_map']),  # Number of questions attempted
                float(performance.get('speed', 0)) if performance.get('speed') else 0
            ])
        
        X = np.array(features)
        
        # Use more sophisticated preprocessing
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Simulated rank generation based on performance
        # This is a proxy since actual ranks aren't available
        y = np.array([
            10000 - (perf['score'] * 100 + float(perf['accuracy'].strip('%')))
            for perf in user_performance_history
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Use Gradient Boosting for regression
        model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42
        )
        
        # Fit and predict
        model.fit(X_train, y_train)
        
        # Predict rank
        latest_performance = X_scaled[-1].reshape(1, -1)
        predicted_rank = model.predict(latest_performance)[0]
        
        # Calculate confidence
        try:
            confidence = model.score(X_test, y_test)
        except Exception:
            confidence = 0.5
        
        return {
            'predicted_rank': max(1, min(10000, int(predicted_rank))),
            'confidence': float(confidence),
            'rank_range': {
                'lower_bound': max(1, int(predicted_rank * 0.9)) if predicted_rank else None,
                'upper_bound': min(10000, int(predicted_rank * 1.1)) if predicted_rank else None
            }
        }

    def analyze_new_user_potential(self, user_id: str) -> dict:
        """
        Provide comprehensive initial guidance for users with no quiz history.
        
        Args:
            user_id: Unique identifier for the student
        
        Returns:
            Detailed initial assessment and recommendations
        """
        # Basic NEET subject areas
        neet_subjects = [
            "Physics", 
            "Chemistry", 
            "Biology", 
            "Zoology", 
            "Botany", 
            "Anatomy", 
            "Physiology"
        ]
        
        # Difficulty progression for new learners
        learning_path = {
            "Foundational": [
                "Basic concepts in Biology",
                "Fundamental Physics principles",
                "Core Chemistry concepts"
            ],
            "Intermediate": [
                "Advanced biological systems",
                "Complex chemical reactions",
                "Advanced physics problem-solving"
            ]
        }
        
        # Diagnostic recommendations
        initial_recommendations = {
            'initial_assessment': {
                'status': 'New Learner',
                'recommended_starting_point': 'Comprehensive NEET preparation strategy'
            },
            'diagnostic_suggestions': [
                "Take a comprehensive diagnostic test to assess current knowledge",
                "Identify baseline understanding across NEET subjects",
                "Create a personalized study roadmap"
            ],
            'initial_focus_areas': neet_subjects,
            'learning_progression': learning_path,
            'study_strategy': {
                'daily_study_hours': '4-6 hours',
                'recommended_resources': [
                    "NCERT textbooks for Physics, Chemistry, and Biology",
                    "Previous years' NEET question papers",
                    "Online video tutorials from reputable coaching platforms"
                ]
            },
            'performance_potential': {
                'initial_rank_projection': {
                    'lower_bound': 5000,
                    'upper_bound': 10000,
                    'confidence': 'Preliminary estimate'
                }
            }
        }
        
        return initial_recommendations

def print_new_user_insights(new_user_analysis):
    """
    Print comprehensive insights for new users.
    """
    import termcolor
    from pyfiglet import figlet_format

    print(termcolor.colored(figlet_format("NEW USER NEET INSIGHTS", font="slant"), "green"))
    
    print("\n🚀 Initial Assessment:")
    print(f"  • Status: {new_user_analysis['initial_assessment']['status']}")
    print(f"  • Recommended Starting Point: {new_user_analysis['initial_assessment']['recommended_starting_point']}")

    print("\n🎯 Diagnostic Suggestions:")
    for suggestion in new_user_analysis['diagnostic_suggestions']:
        print(f"  • {suggestion}")

    print("\n📚 Initial Focus Subjects:")
    for subject in new_user_analysis['initial_focus_areas']:
        print(f"  • {subject}")

    print("\n🧠 Learning Progression:")
    for level, topics in new_user_analysis['learning_progression'].items():
        print(f"\n  {level} Level:")
        for topic in topics:
            print(f"    • {topic}")

    print("\n⏰ Study Strategy:")
    print(f"  • Recommended Daily Study: {new_user_analysis['study_strategy']['daily_study_hours']}")
    
    print("\n📖 Recommended Resources:")
    for resource in new_user_analysis['study_strategy']['recommended_resources']:
        print(f"  • {resource}")

    print("\n📊 Performance Potential:")
    rank_proj = new_user_analysis['performance_potential']['initial_rank_projection']
    print(f"  • Estimated Rank Range: {rank_proj['lower_bound']} - {rank_proj['upper_bound']}")
    print(f"  • Confidence: {rank_proj['confidence']}")

# user_id = "YcDFSO4ZukTJnnFMgRNVwZTE4j42"
# user_id = "7ZXdz3zHuNcdg9agb5YpaOGLQqw2" # new user
    
def print_performance_insights(
    performance_metrics, 
    student_persona, 
    recommendations, 
    neet_rank_prediction
):
    import termcolor
    from pyfiglet import figlet_format

    print(termcolor.colored(figlet_format("NEET PERFORMANCE INSIGHTS", font="slant"), "green"))
    
    # Performance Overview
    print("\n🏆 Performance Overview:")
    print(f"  • Total Quizzes: {performance_metrics.get('total_quizzes', 0)}")
    print(f"  • Average Score: {performance_metrics.get('overall_performance', {}).get('average_score', 0):.2f}")
    print(f"  • Average Accuracy: {performance_metrics.get('overall_performance', {}).get('average_accuracy', 0):.2f}%")

    # Topic Performance
    print("\n📊 Topic Performance Highlights:")
    topic_performance = performance_metrics.get('topic_performance', {})
    
    if topic_performance:
        top_topics = sorted(
            [(topic, details['average_accuracy']) for topic, details in topic_performance.items()],
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        weak_topics = sorted(
            [(topic, details['average_accuracy']) for topic, details in topic_performance.items()],
            key=lambda x: x[1]
        )[:2]

        print("  🟢 Top Performing Topics:")
        for topic, accuracy in top_topics:
            print(f"    • {topic}: {accuracy:.2f}% accuracy")

        print("\n  🔴 Areas for Improvement:")
        for topic, accuracy in weak_topics:
            print(f"    • {topic}: {accuracy:.2f}% accuracy")
    else:
        print("  No topic performance data available")

    # Student Persona
    print("\n🧠 Student Persona:")
    print(f"  Name: {student_persona.get('name', 'Unknown')}")
    print(f"  Description: {student_persona.get('description', 'No specific persona identified')}")

    # NEET Rank Prediction
    print("\n🎯 NEET Rank Prediction:")
    print(f"  • Predicted Rank: {neet_rank_prediction.get('predicted_rank', 'N/A')}")
    print(f"  • Confidence: {neet_rank_prediction.get('confidence', 0) * 100:.2f}%")
    
    rank_range = neet_rank_prediction.get('rank_range', {})
    lower_bound = rank_range.get('lower_bound')
    upper_bound = rank_range.get('upper_bound')
    
    if lower_bound is not None and upper_bound is not None:
        print(f"  • Rank Range: {lower_bound} - {upper_bound}")
    else:
        print("  • Rank Range: Not available")

    # Recommendations
    print("\n💡 Recommendations:")
    weak_topics = recommendations.get('weak_topics', [])
    if weak_topics:
        print("  Focus Areas:")
        for topic in weak_topics:
            print(f"  • {topic}")
    else:
        print("  No specific focus areas identified")
    
    print("\n  Learning Strategy:")
    print(f"  • {recommendations.get('learning_strategy', 'No specific strategy')}")
    
    print("\n  Personalized Tips:")
    tips = recommendations.get('personalized_tips', [])
    if tips:
        for tip in tips:
            print(f"  • {tip}")
    else:
        print("  No specific tips available")

def main():
    analytics = AdvancedNEETStudentAnalytics(
        quiz_endpoints_path='quiz_endpoints.json',
        quiz_submission_path='quiz_submission_data.json',
        historical_quiz_path='historical_quiz_data.json'
    )
    user_id = input("USER ID:")
    
    # Compute analytics
    performance_metrics = analytics.analyze_performance(user_id)
    
    # Check if user has no quiz history
    if performance_metrics['total_quizzes'] == 0:
        new_user_analysis = analytics.analyze_new_user_potential(user_id)
        print_new_user_insights(new_user_analysis)
    else:
        student_persona = analytics.generate_student_persona(performance_metrics)
        recommendations = analytics.recommend_improvement_steps(performance_metrics)
        
        # Rank prediction
        user_history = [quiz for quiz in analytics.historical_quizzes if quiz['user_id'] == user_id]
        neet_rank_prediction = analytics.predict_neet_rank(user_history)
        
        # Performance Insights Visualization
        print_performance_insights(
            performance_metrics, 
            student_persona, 
            recommendations, 
            neet_rank_prediction
        )

if __name__ == "__main__":
    main()
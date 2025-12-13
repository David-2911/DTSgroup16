"""
Generate detailed medical analysis based on classification results.

This module provides comprehensive medical context and explanations
for tumor classification results, making AI predictions more interpretable.
"""
import json
import os


class AnalysisGenerator:
    """
    Generate comprehensive medical analysis for classifications.
    
    Provides detailed explanations, confidence interpretations,
    and educational context for tumor classification results.
    """
    
    def __init__(self, descriptions_path=None):
        """
        Load tumor descriptions database.
        
        Args:
            descriptions_path: Path to tumor descriptions JSON file.
                             If None, uses default relative path.
        """
        if descriptions_path is None:
            # Default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            descriptions_path = os.path.join(
                current_dir, '..', 'shared', 'tumor_descriptions.json'
            )
        
        with open(descriptions_path, 'r') as f:
            self.descriptions = json.load(f)
    
    def generate_analysis(self, predicted_class, confidence, all_probabilities):
        """
        Generate detailed analysis based on prediction.
        
        Args:
            predicted_class: Name of predicted class (e.g., "Glioma")
            confidence: Confidence percentage (0-100)
            all_probabilities: Dict of all class probabilities
            
        Returns:
            dict: Comprehensive analysis with description, interpretation,
                  characteristics, and educational context
        """
        tumor_info = self.descriptions.get(predicted_class, {})
        
        # Build analysis
        analysis = {
            'classification': predicted_class,
            'confidence_level': self._interpret_confidence(confidence),
            'description': tumor_info.get('short_description', 'No description available'),
            'detailed_info': tumor_info.get('detailed_description', ''),
            'characteristics': tumor_info.get('typical_characteristics', []),
            'common_locations': tumor_info.get('common_locations', []),
            'prevalence': tumor_info.get('prevalence', 'Unknown'),
            'model_interpretation': self._generate_model_interpretation(
                predicted_class, 
                confidence, 
                all_probabilities
            ),
            'educational_context': tumor_info.get('educational_note', ''),
            'disclaimer': self._get_disclaimer()
        }
        
        # Add differential diagnosis if confidence is not very high
        if confidence < 90:
            analysis['differential_considerations'] = \
                self._get_differential_diagnosis(all_probabilities, predicted_class)
        
        return analysis
    
    def _interpret_confidence(self, confidence):
        """
        Interpret confidence score into human-readable level.
        
        Args:
            confidence: Percentage (0-100)
            
        Returns:
            str: Confidence interpretation with explanation
        """
        if confidence >= 95:
            return "Very High - Model is very confident in this classification"
        elif confidence >= 85:
            return "High - Model shows strong confidence in this classification"
        elif confidence >= 75:
            return "Moderate - Model is reasonably confident"
        elif confidence >= 60:
            return "Low - Model shows some uncertainty"
        else:
            return "Very Low - Significant uncertainty in classification"
    
    def _generate_model_interpretation(self, predicted_class, confidence, probabilities):
        """
        Generate explanation of what the model detected.
        
        Args:
            predicted_class: Predicted tumor type
            confidence: Confidence percentage
            probabilities: All class probabilities
            
        Returns:
            str: Human-readable model interpretation
        """
        # Sort probabilities
        sorted_probs = sorted(
            probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_class, top_prob = sorted_probs[0]
        second_class, second_prob = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0)
        
        interpretation = f"The model classified this scan as {predicted_class} with {confidence:.1f}% confidence. "
        
        if confidence >= 90:
            interpretation += f"The image features strongly indicate {predicted_class}. "
        elif confidence >= 75:
            interpretation += f"The image shows characteristics consistent with {predicted_class}. "
        else:
            interpretation += f"The image shows some features of {predicted_class}, but "
            if second_prob > 15:
                interpretation += f"also shares characteristics with {second_class} ({second_prob:.1f}%). "
            else:
                interpretation += "with moderate certainty. "
        
        interpretation += "The Grad-CAM heatmap (shown in the visualization) highlights the regions that most influenced this classification."
        
        return interpretation
    
    def _get_differential_diagnosis(self, probabilities, predicted_class):
        """
        Generate differential diagnosis if confidence is not very high.
        
        Args:
            probabilities: All class probabilities
            predicted_class: Top prediction
            
        Returns:
            list: Alternative possibilities with probabilities
        """
        # Get classes with >10% probability
        alternatives = [
            {
                'class': cls,
                'probability': prob,
                'note': f"Consider {cls} - {prob:.1f}% likelihood"
            }
            for cls, prob in probabilities.items()
            if prob > 10 and cls != predicted_class
        ]
        
        # Sort by probability descending
        alternatives.sort(key=lambda x: x['probability'], reverse=True)
        
        return alternatives if alternatives else []
    
    def _get_disclaimer(self):
        """
        Return medical disclaimer.
        
        Returns:
            str: Important disclaimer text for medical context
        """
        return (
            "IMPORTANT: This is a tool with ~92% accuracy on test data. "
            "It is NOT a substitute for professional medical diagnosis. "
            "Always consult qualified healthcare providers for medical advice, "
            "diagnosis, and treatment decisions."
        )
    
    def get_tumor_info(self, tumor_class):
        """
        Get raw tumor information for a specific class.
        
        Args:
            tumor_class: Name of tumor class
            
        Returns:
            dict: Tumor information or empty dict if not found
        """
        return self.descriptions.get(tumor_class, {})

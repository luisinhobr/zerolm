from typing import List
import re

class TemplateValidator:
    """Implements the structural validation from the template"""
    def __init__(self):
        self.required_nodes = [
            ('context_analysis/weighting_system', 0.7),
            ('validation_hierarchy/layer[@type]', 0.9),
            ('attribute_constraints/param[@name]', 0.8)
        ]
        
    def validate_structure(self, xml_tree) -> dict:
        """Robust validation with error handling"""
        try:
            errors = []
            for xpath, _ in self.required_nodes:
                if not xml_tree.find(xpath):
                    errors.append(f"Required element missing: {xpath}")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'confidence': 1.0 - (len(errors) * 0.1)
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'confidence': 0.0
            }

    def enforce_attribute_rules(self, xml_tree) -> List[str]:
        """Enforces attribute format rules from the template"""
        errors = []
        for param in xml_tree.iterfind('.//param'):
            if param.get('name') and not re.match(r'^param_\d+$', param.get('name')):
                errors.append(f"Invalid parameter name: {param.get('name')}")
            if param.get('value') and not re.match(r'^0\.\d{1,2}$', param.get('value')):
                errors.append(f"Invalid value format: {param.get('value')}")
        return errors

from typing import List
import re

class TemplateEnforcer:
    """Enforces template compliance rules"""
    def __init__(self):
        self.allowed_tags = {
            'agent_definition', 'processing_pipeline', 'compliance_system',
            'context_analysis', 'validation_hierarchy', 'attribute_constraints'
        }
        
    def validate_compliance(self, xml_tree) -> List[str]:
        """Enforce template's structural compliance rules"""
        errors = []
        
        # Check for prohibited content
        for element in xml_tree.iter():
            if element.tag not in self.allowed_tags:
                errors.append(f"Prohibited element: {element.tag}")
                
            if element.text and element.text.strip():
                if not any(child.tag in self.allowed_tags for child in element):
                    errors.append(f"Free text block in {element.tag}")
        
        # Validate attribute formats
        for param in xml_tree.iterfind('.//param'):
            if not re.match(r'^0\.\d{1,2}$', param.get('value', '')):
                errors.append(f"Invalid value format: {param.get('value')}")
                
        return errors

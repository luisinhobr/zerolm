from typing import Dict

class AutoCorrector:
    """ImplementaÃ§Ã£o completa do sistema de autocorreÃ§Ã£o"""
    def __init__(self):
        self.correction_layers = [
            {
                'priority': 1,
                'type': 'structural',
                'actions': [
                    self.correct_namespace,
                    self.enforce_component_order,
                    self.validate_numeric_constraints,
                    self.verify_required_elements
                ]
            },
            {
                'priority': 2,
                'type': 'semantic',
                'thresholds': {
                    'value_correction': 0.8,
                    'context_adjustment': 0.7
                }
            }
        ]
        
        # Definir a ordem correta dos componentes
        self.component_order = [
            'context_analysis',
            'validation_hierarchy',
            'attribute_constraints',
            'processing_pipeline'
        ]

    def enforce_component_order(self, xml_tree) -> Dict:
        """Garante a ordem correta dos componentes conforme o template"""
        errors = {}
        current_order = [elem.tag for elem in xml_tree]
        
        for i, expected_tag in enumerate(self.component_order):
            if expected_tag in current_order:
                position = current_order.index(expected_tag)
                if position < i:
                    errors[expected_tag] = {
                        'action': 'move',
                        'from': position,
                        'to': i
                    }
        
        if errors:
            return {
                'name': 'component_order',
                'modified': True,
                'details': errors
            }
        return {
            'name': 'component_order',
            'modified': False
        }

    def validate_numeric_constraints(self, xml_tree) -> Dict:
        """Valida restriÃ§Ãµes numÃ©ricas do template"""
        constraints = {
            './/param[@name]': {
                'min': 0,
                'max': 1,
                'decimal_places': 2
            }
        }
        
        errors = []
        for xpath, rules in constraints.items():
            for element in xml_tree.findall(xpath):
                try:
                    value = float(element.get('value', 0))
                    if not (rules['min'] <= value <= rules['max']):
                        errors.append(f"Valor {value} fora do intervalo em {element.tag}")
                    if len(str(value).split('.')[-1]) > rules['decimal_places']:
                        errors.append(f"PrecisÃ£o excessiva em {element.tag}")
                except ValueError:
                    errors.append(f"Valor nÃ£o numÃ©rico em {element.tag}")
        
        return {
            'name': 'numeric_constraints',
            'modified': bool(errors),
            'details': errors
        }

    def verify_required_elements(self, xml_tree) -> Dict:
        """Verifica elementos obrigatÃ³rios do template"""
        required_elements = [
            'context_analysis/weighting_system',
            'validation_hierarchy/layer',
            'attribute_constraints/param'
        ]
        
        missing = []
        for xpath in required_elements:
            if not xml_tree.find(xpath):
                missing.append(xpath)
        
        return {
            'name': 'required_elements',
            'modified': bool(missing),
            'details': missing
        }

    # MantÃ©m a implementaÃ§Ã£o existente dos outros mÃ©todos
    def correct_namespace(self, xml_tree) -> Dict:
        """Corrige namespace conforme template"""
        if 'xmlns' not in xml_tree.attrib:
            xml_tree.set('xmlns', 'template_v3')
            return {
                'name': 'namespace',
                'modified': True
            }
        return {
            'name': 'namespace',
            'modified': False
        }

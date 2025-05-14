from BaseModelBuilder import BaseModelBuilder

class ModelBuilderVOT(BaseModelBuilder):

    def get_default_model(self):
        p = {}
        p['model'] = self.model_name
        p['train_grid'] = ''
        p['type_ml'] = ''
        
        p['params'] = {}

        return p
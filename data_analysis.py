import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class BlurAnalyzer:
    """Analyze the effect of image blur on classification performance"""
    
    def __init__(self, data_path):
        """Load and prepare data from JSON file"""
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
        
        self.prepare_data()
        
    def prepare_data(self):
        """Extract sigma values and mAP scores for each category"""
        self.sigma_values = []
        self.categories = None
        self.data = {}
        
        for blur_key in sorted(self.raw_data.keys(), 
                              key=lambda x: int(x.replace('blur', ''))):
            blur_data = self.raw_data[blur_key]
            self.sigma_values.append(blur_data['sigma'])
            
            # Initialize categories on first iteration
            if self.categories is None:
                self.categories = list(blur_data['mAP'].keys())
                for cat in self.categories:
                    self.data[cat] = []
            
            # Collect mAP values
            for cat in self.categories:
                self.data[cat].append(blur_data['mAP'][cat])
        
        self.sigma_values = np.array(self.sigma_values)
        for cat in self.categories:
            self.data[cat] = np.array(self.data[cat])
    
    # Regression model functions
    @staticmethod
    def exponential_decay(x, a, b, c):
        """Exponential decay: y = a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c
    
    @staticmethod
    def polynomial_2(x, a, b, c):
        """Quadratic polynomial: y = a + b*x + c*x^2"""
        return a + b * x + c * x**2
    
    @staticmethod
    def polynomial_3(x, a, b, c, d):
        """Cubic polynomial: y = a + b*x + c*x^2 + d*x^3"""
        return a + b * x + c * x**2 + d * x**3
    
    @staticmethod
    def linear(x, a, b):
        """Linear: y = a + b*x"""
        return a + b * x
    
    @staticmethod
    def power_law(x, a, b, c):
        """Power law: y = a * (x + c)^b (for x >= 0)"""
        return a * np.power(x + c, b)
    
    def fit_model(self, category, model_func, initial_guess=None):
        """Fit a regression model to category data"""
        x = self.sigma_values
        y = self.data[category]
        
        try:
            if initial_guess is None:
                # Smart initial guesses based on model type
                if model_func == self.exponential_decay:
                    initial_guess = [y[0] - y[-1], 0.5, y[-1]]
                elif model_func == self.polynomial_2:
                    initial_guess = [y[0], -0.1, 0]
                elif model_func == self.polynomial_3:
                    initial_guess = [y[0], -0.1, 0, 0]
                elif model_func == self.linear:
                    initial_guess = [y[0], -0.1]
                elif model_func == self.power_law:
                    initial_guess = [y[0], -0.5, 0.1]
            
            params, covariance = curve_fit(model_func, x, y, p0=initial_guess, maxfev=10000)
            
            # Calculate predictions and metrics
            y_pred = model_func(x, *params)
            r2 = r2_score(y, y_pred)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            
            # Calculate AIC and BIC
            n = len(y)
            k = len(params)
            rss = np.sum((y - y_pred)**2)
            aic = n * np.log(rss / n) + 2 * k
            bic = n * np.log(rss / n) + k * np.log(n)
            
            return {
                'params': params,
                'covariance': covariance,
                'r2': r2,
                'rmse': rmse,
                'aic': aic,
                'bic': bic,
                'predictions': y_pred
            }
        except Exception as e:
            print(f"Failed to fit {model_func.__name__} for {category}: {str(e)}")
            return None
    
    def analyze_all_models(self):
        """Fit all regression models to all categories"""
        self.results = {}
        
        models = {
            'Exponential Decay': self.exponential_decay,
            'Quadratic': self.polynomial_2,
            'Cubic': self.polynomial_3,
            'Linear': self.linear,
            'Power Law': self.power_law
        }
        
        for category in self.categories:
            self.results[category] = {}
            print(f"\nAnalyzing category: {category}")
            print("=" * 60)
            
            for model_name, model_func in models.items():
                result = self.fit_model(category, model_func)
                if result is not None:
                    self.results[category][model_name] = result
                    print(f"{model_name:20s} | R²={result['r2']:.4f} | RMSE={result['rmse']:.4f} | AIC={result['aic']:.2f}")
    
    def get_best_model(self, category, criterion='r2'):
        """Get the best model for a category based on specified criterion"""
        if category not in self.results:
            return None
        
        best_model = None
        best_value = float('-inf') if criterion == 'r2' else float('inf')
        
        for model_name, result in self.results[category].items():
            if criterion == 'r2':
                if result['r2'] > best_value:
                    best_value = result['r2']
                    best_model = model_name
            elif criterion in ['aic', 'bic', 'rmse']:
                if result[criterion] < best_value:
                    best_value = result[criterion]
                    best_model = model_name
        
        return best_model
    
    def calculate_decay_rates(self):
        """Calculate and compare decay rates between categories"""
        print("\n" + "=" * 80)
        print("DECAY RATE ANALYSIS")
        print("=" * 80)
        
        decay_rates = {}
        
        for category in self.categories:
            # Get exponential decay parameters if available
            if 'Exponential Decay' in self.results[category]:
                params = self.results[category]['Exponential Decay']['params']
                decay_rate = params[1]  # b parameter in a*exp(-b*x) + c
                decay_rates[category] = decay_rate
        
        if decay_rates:
            print("\nExponential Decay Rates (b parameter):")
            print("-" * 60)
            sorted_rates = sorted(decay_rates.items(), key=lambda x: x[1], reverse=True)
            
            for i, (cat, rate) in enumerate(sorted_rates, 1):
                print(f"{i}. {cat:15s}: {rate:.4f} (half-life: σ={np.log(2)/rate:.2f})  (a={self.results[cat]['Exponential Decay']['params'][0]:.4f}, c={self.results[cat]['Exponential Decay']['params'][2]:.4f})")
            
            # Calculate relative decay rates
            print("\nRelative Decay Rates (compared to slowest):")
            print("-" * 60)
            slowest_rate = min(decay_rates.values())
            for cat, rate in sorted_rates:
                relative = rate / slowest_rate
                print(f"{cat:15s}: {relative:.2f}x faster than slowest")
        
        return decay_rates
    
    def find_critical_thresholds(self, threshold=0.5):
        """Find sigma values where mAP drops below threshold"""
        print(f"\n" + "=" * 80)
        print(f"CRITICAL BLUR LEVELS (mAP < {threshold})")
        print("=" * 80)
        
        critical_sigmas = {}
        
        for category in self.categories:
            best_model_name = self.get_best_model(category, 'r2')
            if best_model_name is None:
                continue
            
            model_name_to_func = {
                'Exponential Decay': self.exponential_decay,
                # 'Quadratic': self.polynomial_2,
                # 'Cubic': self.polynomial_3,
                # 'Linear': self.linear,
                # 'Power Law': self.power_law
            }
            
            model_func = model_name_to_func[best_model_name]
            params = self.results[category][best_model_name]['params']
            
            # Search for threshold crossing
            sigma_range = np.linspace(0, max(self.sigma_values) * 2, 1000)
            predictions = model_func(sigma_range, *params)
            
            # Find first sigma where mAP < threshold
            below_threshold = np.where(predictions < threshold)[0]
            if len(below_threshold) > 0:
                critical_sigma = sigma_range[below_threshold[0]]
                critical_sigmas[category] = critical_sigma
                print(f"{category:15s}: σ = {critical_sigma:.2f} ({best_model_name})")
            else:
                print(f"{category:15s}: Never drops below {threshold}")
        
        return critical_sigmas
    
    def plot_results(self, save_path='./da_outputs/blur_analysis.png'):
        """Create comprehensive visualization of results"""
        n_categories = len(self.categories)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Extended sigma range for smooth curves
        sigma_extended = np.linspace(0, max(self.sigma_values) * 1.2, 200)
        
        for idx, category in enumerate(self.categories):
            ax = axes[idx]
            
            # Plot actual data points
            ax.scatter(self.sigma_values, self.data[category], 
                      s=100, color='black', zorder=5, label='Actual Data')
            
            # Plot all fitted models
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for (model_name, result), color in zip(self.results[category].items(), colors):
                model_name_to_func = {
                    'Exponential Decay': self.exponential_decay,
                    # 'Quadratic': self.polynomial_2,
                    # 'Cubic': self.polynomial_3,
                    # 'Linear': self.linear,
                    # 'Power Law': self.power_law
                }
                model_func = model_name_to_func[model_name]
                
                y_pred = model_func(sigma_extended, *result['params'])
                ax.plot(sigma_extended, y_pred, color=color, alpha=0.6, linewidth=2,
                       label=f"{model_name} (R²={result['r2']:.3f})")
            
            # Find and highlight best model
            best_model_name = self.get_best_model(category, 'r2')
            if best_model_name:
                ax.set_title(f"{category.title()}", 
                           fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Blur Sigma (σ)', fontsize=10)
            ax.set_ylabel('mAP', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            ax.set_ylim([0, 1.05])
        
        # Remove extra subplot if odd number of categories
        if n_categories < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        return fig
    
    def generate_summary_table(self, save_path='./da_outputs/summary_table.csv'):
        """Generate summary table of best models and parameters"""
        summary_data = []
        
        for category in self.categories:
            best_model_name = self.get_best_model(category, 'r2')
            if best_model_name is None:
                continue
            
            result = self.results[category][best_model_name]
            
            row = {
                'Category': category,
                'Best_Model': best_model_name,
                'R2': result['r2'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'Initial_mAP': self.data[category][0],
                'Final_mAP': self.data[category][-1],
                'Total_Decline': self.data[category][0] - self.data[category][-1],
                'Percent_Decline': ((self.data[category][0] - self.data[category][-1]) / 
                                   self.data[category][0] * 100) if self.data[category][0] > 0 else 0
            }
            
            # Add model-specific parameters
            for i, param in enumerate(result['params']):
                row[f'Param_{i+1}'] = param
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(save_path, index=False)
        print(f"\nSummary table saved to: {save_path}")
        print("\nSummary Statistics:")
        print("=" * 80)
        print(df.to_string(index=False))
        
        return df
    
    def generate_report(self):
        """Generate a complete analysis report"""
        print("\n" + "=" * 80)
        print("BLUR EFFECT ANALYSIS REPORT")
        print("=" * 80)
        print(f"\nDataset: {len(self.sigma_values)} blur levels")
        print(f"Sigma range: {min(self.sigma_values):.2f} to {max(self.sigma_values):.2f}")
        print(f"Categories analyzed: {', '.join(self.categories)}")
        
        # Perform all analyses
        self.analyze_all_models()
        decay_rates = self.calculate_decay_rates()
        critical_sigmas = self.find_critical_thresholds(0.5)
        
        # Generate outputs
        self.plot_results()
        self.generate_summary_table()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

    def run_exponential(self):
        """Fit exponential regression models to all categories"""
        self.results = {}
        
        models = {
            'Exponential Decay': self.exponential_decay,
        }
        
        for category in self.categories:
            self.results[category] = {}
            print(f"\nAnalyzing category: {category}")
            print("=" * 60)
            
            for model_name, model_func in models.items():
                result = self.fit_model(category, model_func)
                if result is not None:
                    self.results[category][model_name] = result
                    print(f"{model_name:20s} | R²={result['r2']:.4f} | RMSE={result['rmse']:.4f} | AIC={result['aic']:.2f}")


# Main execution
if __name__ == "__main__":
    import sys
    
    # Check if data file is provided
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = 'training_prediction_scores.json'
    
    print(f"Loading data from: {data_file}")
    
    try:
        analyzer = BlurAnalyzer(data_file)
        # analyzer.generate_report() # for running across all regression types

        analyzer.prepare_data()
        analyzer.run_exponential()
        analyzer.calculate_decay_rates()
        analyzer.find_critical_thresholds()
        analyzer.plot_results()
    except FileNotFoundError:
        print(f"\nError: Could not find data file at {data_file}")
        print("Please provide the path to your JSON data file as an argument.")
        print("Example: python blur_analysis.py /path/to/your/data.json")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

    

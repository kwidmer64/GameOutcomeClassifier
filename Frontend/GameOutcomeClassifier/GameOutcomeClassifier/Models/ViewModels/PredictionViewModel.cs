namespace GameOutcomeClassifier.Models.ViewModels
{
    public class PredictionViewModel
    {
        public Dictionary<string, string> Teams { get; set; } = new();
        public PredictionResult? Result { get; set; }
    }
}

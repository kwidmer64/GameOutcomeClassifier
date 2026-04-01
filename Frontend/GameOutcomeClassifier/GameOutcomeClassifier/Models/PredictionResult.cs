namespace GameOutcomeClassifier.Models
{
    public class PredictionResult
    {
        public string Winner { get; set; } = "";
        public double Confidence { get; set; }
        public string HomeTeam { get; set; } = "";
        public string AwayTeam { get; set; } = "";
        public Dictionary<string, double> HomeScoreProbabilities { get; set; } = new Dictionary<string, double>();
        public Dictionary<string, double> AwayScoreProbabilities { get; set; } = new Dictionary<string, double>();
    }
}

namespace GameOutcomeClassifier.Models
{
    public class MatchupRequest
    {
        public string HomeTeam { get; set; } = "";
        public string AwayTeam { get; set; } = "";

        public MatchupRequest(string homeTeam, string awayTeam)
        {
            HomeTeam = homeTeam;
            AwayTeam = awayTeam;
        }
    }
}

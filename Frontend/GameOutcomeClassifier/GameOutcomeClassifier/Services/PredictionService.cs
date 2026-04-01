using GameOutcomeClassifier.Models;
using System.Text.Json;

namespace GameOutcomeClassifier.Services
{
    public class PredictionService
    {
        private readonly HttpClient _httpClient;

        public PredictionService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<PredictionResult?> PostPredictionAsync(string home, string away)
        {
            // Create matchup request with home and away teams
            MatchupRequest request = new(home, away);

            // Tell the serializer to deal with the snake case coming from the API
            JsonSerializerOptions options = new()
            {
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            };

            // POST request to the backend
            HttpResponseMessage response = await _httpClient.PostAsJsonAsync("/predict", request, options);

            response.EnsureSuccessStatusCode();

            // Read and return the response body
            // Response body is of type PredictionResult?
            var result = await response.Content.ReadFromJsonAsync<PredictionResult>(options);
            return result;
        }

        public async Task<Dictionary<string, string>> GetTeamsAsync()
        {
            // Tell the serializer to deal with the snake case coming from the API
            JsonSerializerOptions options = new()
            {
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            };

            // Call the GET /teams endpoint
            var response = await _httpClient.GetFromJsonAsync<Dictionary<string, string>>("/teams", options);

            // Check if the response was null
            if (response == null)
            {
                throw new Exception("Could not retrieve teams. Possible null response.");
            }

            return response;
        }
    }
}

using GameOutcomeClassifier.Models;
using GameOutcomeClassifier.Models.ViewModels;
using GameOutcomeClassifier.Services;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;

namespace GameOutcomeClassifier.Controllers
{
    public class HomeController : Controller
    {
        private readonly PredictionService _predictionService;

        public HomeController(PredictionService predictionService)
        {
            _predictionService = predictionService;
        }

        public async Task<IActionResult> Index()
        {
            PredictionViewModel predictionViewModel = new();

            // Retrieve a list of teams from the backend
            Dictionary<string, string> teams = await _predictionService.GetTeamsAsync();
            predictionViewModel.Teams = teams;

            return View(predictionViewModel);
        }

        [HttpPost]
        public async Task<IActionResult> Predict(string homeTeam, string awayTeam)
        {
            PredictionViewModel predictionViewModel = new();

            // POST to predict the outcome
            PredictionResult? result = await _predictionService.PostPredictionAsync(homeTeam, awayTeam);
            Dictionary<string, string> teams = await _predictionService.GetTeamsAsync();

            // Set view model properties
            predictionViewModel.Teams = teams;
            predictionViewModel.Result = result;

            // Check if the result is null
            if (result == null)
            {
                // TODO: Add error message to page
                return View("Index");
            }

            return View("Index", predictionViewModel);
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}

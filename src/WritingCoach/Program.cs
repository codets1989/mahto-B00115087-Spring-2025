

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddHttpClient(); 
builder.Services.AddCors(options =>
{
    options.AddPolicy("ReactFrontend", policy =>
    {
        policy.WithOrigins("http://localhost:5173")
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});
var app = builder.Build();
app.UseCors("ReactFrontend");
app.MapPost("/api/grammar/correct", async (HttpClient httpClient, GrammarRequest request) =>
{
    var pythonApiUrl = "http://localhost:8000/correct";
    var response = await httpClient.PostAsJsonAsync(pythonApiUrl, request);
    return await response.Content.ReadFromJsonAsync<dynamic>();
});
app.MapPost("/api/grammar/check_grammar", async (HttpClient httpClient, GrammarRequest request) =>
{
    var pythonApiUrl = "http://localhost:8000/check_grammar";
    var response = await httpClient.PostAsJsonAsync(pythonApiUrl, request);
    return await response.Content.ReadFromJsonAsync<dynamic>();
});
app.Run();
public class GrammarRequest 
{
    public required string Text { get; set; }
}
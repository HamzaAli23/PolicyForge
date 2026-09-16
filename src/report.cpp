#include "policyforge/report.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace policyforge {
namespace {

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << character;
        }
    }
    return output.str();
}

void ensure_stream(const std::ofstream& stream, const std::filesystem::path& path) {
    if (!stream) throw std::runtime_error("Unable to write " + path.string());
}

std::vector<double> moving_average(const std::vector<double>& values, std::size_t window) {
    std::vector<double> result;
    if (values.empty()) return result;
    for (std::size_t start = 0; start < values.size(); start += window) {
        const std::size_t end = std::min(values.size(), start + window);
        const double sum = std::accumulate(values.begin() + static_cast<std::ptrdiff_t>(start),
                                           values.begin() + static_cast<std::ptrdiff_t>(end), 0.0);
        result.push_back(sum / static_cast<double>(end - start));
    }
    return result;
}

std::string polyline(const std::vector<double>& values, int width, int height) {
    if (values.empty()) return {};
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    const double span = std::max(1e-9, *maximum - *minimum);
    std::ostringstream points;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double x = values.size() == 1 ? 0 : static_cast<double>(index) * width / (values.size() - 1);
        const double y = height - (values[index] - *minimum) * height / span;
        if (index) points << ' ';
        points << std::fixed << std::setprecision(1) << x << ',' << y;
    }
    return points.str();
}

std::string policy_grid(const GridWorld& environment, const std::vector<Action>& policy) {
    std::ostringstream html;
    html << "<div class=policy-grid style=\"grid-template-columns:repeat(" << environment.width() << ",1fr)\">";
    for (int y = 0; y < environment.height(); ++y) {
        for (int x = 0; x < environment.width(); ++x) {
            const std::size_t state = environment.index({x, y});
            if (environment.is_wall(state)) html << "<div class=cell-wall>wall</div>";
            else if (const auto reward = environment.terminal_reward(state)) {
                html << "<div class=\"cell-terminal " << (reward.value() >= 0 ? "positive" : "negative") << "\">"
                     << (reward.value() >= 0 ? "+" : "") << reward.value() << "</div>";
            } else {
                html << "<div class=cell-action><span>" << action_symbol(policy[state]) << "</span><small>" << x << ',' << y << "</small></div>";
            }
        }
    }
    return html.str() + "</div>";
}

}  // namespace

void write_report(const GridWorld& environment,
                  const ExperimentReport& report,
                  const std::filesystem::path& output_directory,
                  const std::filesystem::path& environment_file) {
    std::filesystem::create_directories(output_directory);

    const auto summary_path = output_directory / "summary.json";
    std::ofstream summary(summary_path);
    ensure_stream(summary, summary_path);
    summary << std::fixed << std::setprecision(4)
            << "{\n"
            << "  \"environment\": \"" << json_escape(environment_file.generic_string()) << "\",\n"
            << "  \"seed\": " << report.q_config.seed << ",\n"
            << "  \"trainingEpisodes\": " << report.q_config.episodes << ",\n"
            << "  \"evaluationEpisodes\": " << report.learned_evaluation.episodes << ",\n"
            << "  \"maxSteps\": " << report.q_config.max_steps << ",\n"
            << "  \"evaluationWorkers\": " << report.evaluation_workers << ",\n"
            << "  \"discountFactor\": " << report.q_config.gamma << ",\n"
            << "  \"learningRate\": " << report.q_config.alpha << ",\n"
            << "  \"valueIterationIterations\": " << report.optimal.iterations << ",\n"
            << "  \"policyAgreementPercent\": " << report.agreement_percent << ",\n"
            << "  \"optimalPolicy\": {\"successRatePercent\": " << report.optimal_evaluation.success_rate
            << ", \"averageReturn\": " << report.optimal_evaluation.average_return
            << ", \"averageSteps\": " << report.optimal_evaluation.average_steps << "},\n"
            << "  \"learnedPolicy\": {\"successRatePercent\": " << report.learned_evaluation.success_rate
            << ", \"averageReturn\": " << report.learned_evaluation.average_return
            << ", \"averageSteps\": " << report.learned_evaluation.average_steps << "}\n"
            << "}\n";

    const auto csv_path = output_directory / "training.csv";
    std::ofstream csv(csv_path);
    ensure_stream(csv, csv_path);
    csv << "episode,discounted_return,steps\n";
    for (std::size_t episode = 0; episode < report.learned.episode_returns.size(); ++episode) {
        csv << (episode + 1) << ',' << std::setprecision(10) << report.learned.episode_returns[episode]
            << ',' << report.learned.episode_steps[episode] << '\n';
    }

    const auto optimal_path = output_directory / "optimal-policy.txt";
    std::ofstream optimal_file(optimal_path);
    ensure_stream(optimal_file, optimal_path);
    optimal_file << environment.render_policy(report.optimal.policy);

    const auto learned_path = output_directory / "learned-policy.txt";
    std::ofstream learned_file(learned_path);
    ensure_stream(learned_file, learned_path);
    learned_file << environment.render_policy(report.learned.policy);

    const auto averages = moving_average(report.learned.episode_returns, 50);
    const auto points = polyline(averages, 720, 210);
    const auto report_path = output_directory / "report.html";
    std::ofstream html(report_path);
    ensure_stream(html, report_path);
    html << std::fixed << std::setprecision(2);
    html << R"HTML(<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PolicyForge Experiment</title><style>
:root{--bg:#070b0d;--surface:#0e1417;--surface-2:#141c20;--line:#29343a;--text:#eef4f1;--muted:#8d9b97;--lime:#c8ff4d;--cyan:#56e6ff;--orange:#ff7a45;--red:#ff4f64}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:"IBM Plex Mono","Cascadia Code",Consolas,monospace}.page{min-height:100vh;background:radial-gradient(circle at 78% 12%,rgba(86,230,255,.08),transparent 25%),linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:auto,42px 42px,42px 42px}.topbar{height:68px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 4vw;background:rgba(7,11,13,.9);position:sticky;top:0;z-index:2;backdrop-filter:blur(14px)}.brand{font-size:18px;font-weight:900;letter-spacing:-.8px}.brand b{color:var(--lime)}.run-id,.status{font-size:10px;letter-spacing:1.3px;text-transform:uppercase;color:var(--muted)}.status{color:var(--lime)}.status:before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--lime);box-shadow:0 0 14px var(--lime);margin-right:9px}.main{width:min(1480px,92vw);margin:auto;padding:72px 0 80px}.hero{display:grid;grid-template-columns:1.35fr .65fr;gap:60px;align-items:end;padding-bottom:55px;border-bottom:1px solid var(--line)}.kicker{font-size:11px;color:var(--cyan);letter-spacing:2px;text-transform:uppercase}.hero h1{font-family:Arial,sans-serif;font-size:clamp(52px,7vw,104px);letter-spacing:-5px;line-height:.88;margin:22px 0 25px;max-width:850px}.hero p{color:var(--muted);font-size:13px;line-height:1.8;max-width:700px}.signal{text-align:right}.signal strong{display:block;font-family:Arial,sans-serif;color:var(--lime);font-size:clamp(62px,8vw,118px);letter-spacing:-6px;line-height:.8}.signal span{display:block;color:var(--muted);font-size:10px;letter-spacing:1.3px;text-transform:uppercase;margin-top:18px}.telemetry{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--line)}.metric{padding:27px 24px;border-right:1px solid var(--line)}.metric:last-child{border-right:0}.metric small{display:block;color:var(--muted);font-size:9px;letter-spacing:1.3px;text-transform:uppercase}.metric strong{display:block;font-size:25px;margin-top:10px}.metric em{font-style:normal;color:#60706c;font-size:9px}.section-head{display:flex;justify-content:space-between;align-items:end;margin:72px 0 22px}.section-head h2{font-family:Arial,sans-serif;font-size:30px;letter-spacing:-1px;margin:0}.section-head p{margin:0;color:var(--muted);font-size:10px}.index{color:var(--lime);font-size:11px;margin-right:12px}.chart{border:1px solid var(--line);background:linear-gradient(180deg,rgba(86,230,255,.05),transparent);padding:26px;position:relative}.chart:before{content:"RETURN / 50 EPISODES";position:absolute;top:15px;left:18px;color:#59706f;font-size:8px;letter-spacing:1px}.chart svg{display:block;width:100%;height:300px;margin-top:12px}.policies{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line)}.policy{background:var(--surface);padding:28px}.policy-label{display:flex;justify-content:space-between;margin-bottom:22px}.policy-label h3{margin:0;font-size:13px;text-transform:uppercase;letter-spacing:1px}.policy-label span{font-size:9px;color:var(--muted)}.policy-grid{display:grid;gap:5px}.policy-grid>div{min-height:62px;display:grid;place-items:center;font-weight:900;border:1px solid #263237}.cell-action{background:#111a1d;color:var(--cyan);position:relative;font-size:21px}.cell-action small{position:absolute;bottom:5px;right:6px;color:#53625f;font-size:7px}.cell-wall{background:repeating-linear-gradient(135deg,#1c2529,#1c2529 6px,#252f34 6px,#252f34 12px);color:#7b8985;font-size:8px;text-transform:uppercase}.cell-terminal{color:#071012;font-family:Arial,sans-serif;font-size:16px}.cell-terminal.positive{background:var(--lime)}.cell-terminal.negative{background:var(--orange)}.evaluation{display:grid;grid-template-columns:.55fr 1.45fr;border:1px solid var(--line);background:var(--surface)}.evaluation-copy{padding:30px;border-right:1px solid var(--line)}.evaluation-copy h2{font-family:Arial,sans-serif;font-size:29px;letter-spacing:-1px;margin:8px 0 16px}.evaluation-copy p{color:var(--muted);font-size:10px;line-height:1.7}.compare{width:100%;border-collapse:collapse;font-size:12px}.compare th,.compare td{text-align:left;padding:21px 18px;border-bottom:1px solid var(--line)}.compare tr:last-child td{border-bottom:0}.compare th{color:var(--muted);font-size:8px;text-transform:uppercase;letter-spacing:1px}.compare td:first-child{color:var(--cyan);font-weight:800}.note{margin-top:26px;border-left:3px solid var(--orange);padding:13px 16px;background:rgba(255,122,69,.06);color:#a5b2ae;font-size:10px;line-height:1.6}.footer{display:flex;justify-content:space-between;border-top:1px solid var(--line);margin-top:70px;padding-top:20px;color:#54625f;font-size:9px;text-transform:uppercase;letter-spacing:1px}@media(max-width:900px){.main{padding-top:42px}.run-id{display:none}.hero{grid-template-columns:1fr}.signal{text-align:left}.telemetry{grid-template-columns:repeat(2,1fr)}.metric:nth-child(2){border-right:0}.policies,.evaluation{grid-template-columns:1fr}.evaluation-copy{border-right:0;border-bottom:1px solid var(--line)}.section-head{align-items:start;gap:15px;flex-direction:column}.topbar{padding:0 5vw}}
</style></head><body><div class="page"><header class="topbar"><div class="brand">POLICY<b>FORGE</b></div><div class="run-id">EXP_2026 / STOCHASTIC GRID / C++20</div><div class="status">run complete</div></header><main class="main"><section class="hero"><div><div class="kicker">Reinforcement learning field report</div><h1>Learn.<br>Compare.<br>Decide.</h1><p>A model-free Q-learning agent is measured against a dynamic-programming reference under the same stochastic transition model.</p></div><div class="signal"><strong>)HTML";
    html << report.agreement_percent << "%</strong><span>policy agreement / non-terminal states</span></div></section><section class=telemetry>";
    html << "<article class=metric><small>Training volume</small><strong>" << report.q_config.episodes << "</strong><em>episodes</em></article>";
    html << "<article class=metric><small>Learned success</small><strong>" << report.learned_evaluation.success_rate << "%</strong><em>held-out rollouts</em></article>";
    html << "<article class=metric><small>Mean return</small><strong>" << report.learned_evaluation.average_return << "</strong><em>discounted</em></article>";
    html << "<article class=metric><small>Experiment seed</small><strong>" << report.q_config.seed << "</strong><em>reproducible</em></article></section>";
    html << "<div class=section-head><h2><span class=index>01</span>Learning signal</h2><p>Mean discounted return for each block of 50 episodes</p></div><div class=chart><svg viewBox=\"0 0 720 210\" preserveAspectRatio=\"none\"><defs><linearGradient id=area x1=0 y1=0 x2=0 y2=1><stop offset=0 stop-color=\"#56e6ff\" stop-opacity=.30/><stop offset=1 stop-color=\"#56e6ff\" stop-opacity=.01/></linearGradient></defs><line x1=0 y1=209 x2=720 y2=209 stroke=\"#29343a\"/><polyline points=\"" << points << " 720,210 0,210\" fill=\"url(#area)\" stroke=\"none\"/><polyline points=\"" << points << "\" fill=\"none\" stroke=\"#c8ff4d\" stroke-width=2 vector-effect=\"non-scaling-stroke\"/></svg></div>";
    html << "<div class=section-head><h2><span class=index>02</span>Policy blueprints</h2><p>Reference planning vs sampled experience</p></div><section class=policies><article class=policy><div class=policy-label><h3>Value iteration</h3><span>known transition model</span></div>" << policy_grid(environment, report.optimal.policy) << "</article>";
    html << "<article class=policy><div class=policy-label><h3>Q-learning</h3><span>sampled experience only</span></div>" << policy_grid(environment, report.learned.policy) << "</article></section>";
    html << "<div class=section-head><h2><span class=index>03</span>Held-out evaluation</h2><p>Identical episode and step limits</p></div><section class=evaluation><div class=evaluation-copy><span class=kicker>Final comparison</span><h2>Two routes.<br>One objective.</h2><p>The reference maximizes expected discounted return. The learned agent selected a longer, safer route in this seeded experiment.</p></div><div><table class=compare><thead><tr><th>Policy</th><th>Success rate</th><th>Average return</th><th>Average steps</th></tr></thead><tbody>";
    html << "<tr><td>Value iteration</td><td>" << report.optimal_evaluation.success_rate << "%</td><td>" << report.optimal_evaluation.average_return << "</td><td>" << report.optimal_evaluation.average_steps << "</td></tr>";
    html << "<tr><td>Q-learning</td><td>" << report.learned_evaluation.success_rate << "%</td><td>" << report.learned_evaluation.average_return << "</td><td>" << report.learned_evaluation.average_steps << "</td></tr></tbody></table><div class=note>The bundled environment is a fixed regression scenario with stochastic transitions. Results describe this controlled experiment, not performance in an external physical task.</div></div></section><footer class=footer><span>PolicyForge / local experiment artifact</span><span>C++20 · value iteration · Q-learning · parallel evaluation</span></footer></main></div></body></html>";
}

}  // namespace policyforge

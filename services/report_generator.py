"""
Report Generator Service.

Generates PowerPoint reports for creative testing results using pptxgenjs.

Slides:
1. Title slide with campaign info
2. Creatives tested (overview table)
3. Results summary (pass/fail table)
4. Deep dive on what didn't work
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from models import TestResults, CampaignRecommendations, Campaign


# Color palette (no # prefix for pptxgenjs)
COLORS = {
    "primary": "1E3A5F",      # Dark blue
    "secondary": "4A90A4",    # Teal
    "accent": "2ECC71",       # Green for pass
    "warning": "E74C3C",      # Red for fail
    "light": "F5F7FA",        # Light background
    "dark": "2C3E50",         # Dark text
    "white": "FFFFFF",
}


class ReportGenerator:
    """
    Generates PowerPoint reports for creative testing results.
    
    Uses pptxgenjs via Node.js for reliable PPT generation.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory for output files. Defaults to temp dir.
        """
        self.output_dir = output_dir or tempfile.gettempdir()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        results: TestResults,
        recommendations: CampaignRecommendations,
        campaign_name: str = None,
        brand_name: str = None,
    ) -> str:
        """
        Generate a PowerPoint report.
        
        Args:
            results: Test results data
            recommendations: Analysis recommendations
            campaign_name: Optional campaign name override
            brand_name: Optional brand name
            
        Returns:
            Path to generated PPT file
        """
        campaign_name = campaign_name or f"Campaign {results.campaign_id}"
        brand_name = brand_name or "Brand"
        
        # Build the JavaScript code for pptxgenjs
        js_code = self._build_pptx_script(
            results=results,
            recommendations=recommendations,
            campaign_name=campaign_name,
            brand_name=brand_name,
        )
        
        # Write JS to temp file
        js_path = os.path.join(self.output_dir, "generate_report.js")
        with open(js_path, "w") as f:
            f.write(js_code)
        
        # Run with Node.js
        output_path = os.path.join(
            self.output_dir, 
            f"CT_Report_{results.campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
        )
        
        try:
            result = subprocess.run(
                ["node", js_path],
                capture_output=True,
                text=True,
                cwd=self.output_dir,
                timeout=60,
            )
            
            if result.returncode != 0:
                raise Exception(f"Node.js error: {result.stderr}")
            
            return output_path
            
        except FileNotFoundError:
            raise Exception("Node.js not found. Please install Node.js and pptxgenjs.")
        except subprocess.TimeoutExpired:
            raise Exception("Report generation timed out")
    
    def _build_pptx_script(
        self,
        results: TestResults,
        recommendations: CampaignRecommendations,
        campaign_name: str,
        brand_name: str,
    ) -> str:
        """Build the JavaScript code for generating the PPT."""
        
        # Prepare data for JavaScript
        creatives_tested = []
        for r in results.results:
            creatives_tested.append({
                "name": r.creative_name,
                "type": r.asset_type.value.title(),
                "passed": r.passed,
                "lift": f"{r.primary_kpi_lift:.1f}%",
                "stat_sig": "Yes" if r.primary_kpi_stat_sig else "No",
            })
        
        failed_creatives = []
        for rec in recommendations.recommendations:
            if rec.recommendation != "run":
                failed_creatives.append({
                    "name": rec.creative_name,
                    "issues": rec.diagnostic_insights[:3],
                    "improvements": rec.suggested_improvements[:3],
                })
        
        # Escape data for JavaScript
        campaign_name_js = json.dumps(campaign_name)
        brand_name_js = json.dumps(brand_name)
        creatives_json = json.dumps(creatives_tested)
        failed_json = json.dumps(failed_creatives)
        
        output_path = os.path.join(
            self.output_dir,
            f"CT_Report_{results.campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
        )
        output_path_js = json.dumps(output_path)
        
        js_code = f'''
const pptxgen = require("pptxgenjs");

// Data
const campaignName = {campaign_name_js};
const brandName = {brand_name_js};
const creativesData = {creatives_json};
const failedData = {failed_json};
const passRate = {results.pass_rate * 100:.0f};
const totalTested = {results.total_creatives_tested};
const passed = {results.creatives_passed};
const failed = {results.creatives_failed};

// Create presentation
let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Creative Testing Results";
pres.author = "CT Orchestrator";

// ===========================================
// SLIDE 1: Title Slide
// ===========================================
let slide1 = pres.addSlide();
slide1.background = {{ color: "{COLORS['primary']}" }};

// Title
slide1.addText("Creative Testing Results", {{
    x: 0.5, y: 1.5, w: 9, h: 1,
    fontSize: 44, fontFace: "Arial", bold: true,
    color: "{COLORS['white']}"
}});

// Campaign name
slide1.addText(campaignName, {{
    x: 0.5, y: 2.6, w: 9, h: 0.6,
    fontSize: 28, fontFace: "Arial",
    color: "{COLORS['secondary']}"
}});

// Brand and date
slide1.addText(brandName + " | " + new Date().toLocaleDateString(), {{
    x: 0.5, y: 4.5, w: 9, h: 0.4,
    fontSize: 16, fontFace: "Arial",
    color: "{COLORS['white']}"
}});

// Pass rate callout
slide1.addShape(pres.shapes.RECTANGLE, {{
    x: 7, y: 1.5, w: 2.5, h: 2,
    fill: {{ color: "{COLORS['white']}", transparency: 10 }}
}});

slide1.addText(passRate + "%", {{
    x: 7, y: 1.7, w: 2.5, h: 1,
    fontSize: 48, fontFace: "Arial", bold: true,
    color: "{COLORS['white']}", align: "center"
}});

slide1.addText("Pass Rate", {{
    x: 7, y: 2.8, w: 2.5, h: 0.5,
    fontSize: 14, fontFace: "Arial",
    color: "{COLORS['white']}", align: "center"
}});

// ===========================================
// SLIDE 2: Creatives Tested
// ===========================================
let slide2 = pres.addSlide();
slide2.background = {{ color: "{COLORS['light']}" }};

slide2.addText("Creatives Tested", {{
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontSize: 32, fontFace: "Arial", bold: true,
    color: "{COLORS['dark']}"
}});

// Summary stats
slide2.addText(totalTested + " Creatives | " + passed + " Passed | " + failed + " Failed", {{
    x: 0.5, y: 0.9, w: 9, h: 0.4,
    fontSize: 16, fontFace: "Arial",
    color: "{COLORS['secondary']}"
}});

// Table of creatives
let tableData2 = [
    [
        {{ text: "Creative", options: {{ bold: true, fill: {{ color: "{COLORS['primary']}" }}, color: "{COLORS['white']}" }} }},
        {{ text: "Type", options: {{ bold: true, fill: {{ color: "{COLORS['primary']}" }}, color: "{COLORS['white']}" }} }},
        {{ text: "Lift", options: {{ bold: true, fill: {{ color: "{COLORS['primary']}" }}, color: "{COLORS['white']}" }} }},
        {{ text: "Stat Sig", options: {{ bold: true, fill: {{ color: "{COLORS['primary']}" }}, color: "{COLORS['white']}" }} }},
        {{ text: "Result", options: {{ bold: true, fill: {{ color: "{COLORS['primary']}" }}, color: "{COLORS['white']}" }} }}
    ]
];

creativesData.forEach(c => {{
    tableData2.push([
        c.name,
        c.type,
        c.lift,
        c.stat_sig,
        {{ 
            text: c.passed ? "PASS" : "FAIL", 
            options: {{ 
                bold: true,
                color: c.passed ? "{COLORS['accent']}" : "{COLORS['warning']}"
            }} 
        }}
    ]);
}});

slide2.addTable(tableData2, {{
    x: 0.5, y: 1.5, w: 9, h: 3.5,
    colW: [3, 1.2, 1.2, 1.2, 1.2],
    border: {{ pt: 0.5, color: "CCCCCC" }},
    fontFace: "Arial",
    fontSize: 12
}});

// ===========================================
// SLIDE 3: Results Summary
// ===========================================
let slide3 = pres.addSlide();
slide3.background = {{ color: "{COLORS['light']}" }};

slide3.addText("Results Summary", {{
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontSize: 32, fontFace: "Arial", bold: true,
    color: "{COLORS['dark']}"
}});

// Pass card
slide3.addShape(pres.shapes.RECTANGLE, {{
    x: 0.5, y: 1.2, w: 4, h: 2.5,
    fill: {{ color: "{COLORS['accent']}" }}
}});

slide3.addText(passed.toString(), {{
    x: 0.5, y: 1.5, w: 4, h: 1.2,
    fontSize: 72, fontFace: "Arial", bold: true,
    color: "{COLORS['white']}", align: "center"
}});

slide3.addText("Creatives Passed", {{
    x: 0.5, y: 2.8, w: 4, h: 0.5,
    fontSize: 18, fontFace: "Arial",
    color: "{COLORS['white']}", align: "center"
}});

slide3.addText("Recommended to run in campaign", {{
    x: 0.5, y: 3.3, w: 4, h: 0.3,
    fontSize: 12, fontFace: "Arial",
    color: "{COLORS['white']}", align: "center"
}});

// Fail card
slide3.addShape(pres.shapes.RECTANGLE, {{
    x: 5.5, y: 1.2, w: 4, h: 2.5,
    fill: {{ color: "{COLORS['warning']}" }}
}});

slide3.addText(failed.toString(), {{
    x: 5.5, y: 1.5, w: 4, h: 1.2,
    fontSize: 72, fontFace: "Arial", bold: true,
    color: "{COLORS['white']}", align: "center"
}});

slide3.addText("Creatives Failed", {{
    x: 5.5, y: 2.8, w: 4, h: 0.5,
    fontSize: 18, fontFace: "Arial",
    color: "{COLORS['white']}", align: "center"
}});

slide3.addText("Not recommended - see deep dive", {{
    x: 5.5, y: 3.3, w: 4, h: 0.3,
    fontSize: 12, fontFace: "Arial",
    color: "{COLORS['white']}", align: "center"
}});

// Key insight
slide3.addShape(pres.shapes.RECTANGLE, {{
    x: 0.5, y: 4.2, w: 9, h: 1,
    fill: {{ color: "{COLORS['primary']}" }}
}});

slide3.addText("Key Insight: " + (passRate >= 50 ? 
    "Majority of creatives are performing well" : 
    "Creative strategy may need review - low pass rate"), {{
    x: 0.7, y: 4.4, w: 8.6, h: 0.6,
    fontSize: 16, fontFace: "Arial",
    color: "{COLORS['white']}"
}});

// ===========================================
// SLIDE 4: Deep Dive - What Didn't Work
// ===========================================
let slide4 = pres.addSlide();
slide4.background = {{ color: "{COLORS['light']}" }};

slide4.addText("Deep Dive: What Didn't Work", {{
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontSize: 32, fontFace: "Arial", bold: true,
    color: "{COLORS['dark']}"
}});

if (failedData.length === 0) {{
    slide4.addText("All creatives passed! No issues to report.", {{
        x: 0.5, y: 2, w: 9, h: 1,
        fontSize: 24, fontFace: "Arial",
        color: "{COLORS['accent']}", align: "center"
    }});
}} else {{
    let yPos = 1.2;
    
    failedData.slice(0, 3).forEach((creative, idx) => {{
        // Creative name header
        slide4.addShape(pres.shapes.RECTANGLE, {{
            x: 0.5, y: yPos, w: 0.1, h: 1.2,
            fill: {{ color: "{COLORS['warning']}" }}
        }});
        
        slide4.addText(creative.name, {{
            x: 0.8, y: yPos, w: 4, h: 0.4,
            fontSize: 16, fontFace: "Arial", bold: true,
            color: "{COLORS['dark']}"
        }});
        
        // Issues
        let issueText = creative.issues.length > 0 ? 
            "Issues: " + creative.issues.join("; ") : 
            "No specific diagnostic issues identified";
        
        slide4.addText(issueText, {{
            x: 0.8, y: yPos + 0.4, w: 8.5, h: 0.4,
            fontSize: 11, fontFace: "Arial",
            color: "{COLORS['dark']}"
        }});
        
        // Improvements
        let improveText = creative.improvements.length > 0 ?
            "Suggested: " + creative.improvements.join("; ") :
            "Consider alternative creative approach";
        
        slide4.addText(improveText, {{
            x: 0.8, y: yPos + 0.8, w: 8.5, h: 0.4,
            fontSize: 11, fontFace: "Arial", italic: true,
            color: "{COLORS['secondary']}"
        }});
        
        yPos += 1.4;
    }});
}}

// Save presentation
pres.writeFile({{ fileName: {output_path_js} }})
    .then(() => console.log("Report generated successfully"))
    .catch(err => {{ console.error(err); process.exit(1); }});
'''
        
        return js_code
    
    def generate_simple_report(
        self,
        results: TestResults,
        recommendations: CampaignRecommendations,
        campaign_name: str = None,
    ) -> str:
        """
        Generate a simple text-based report (fallback if Node.js unavailable).
        
        Returns markdown formatted report.
        """
        campaign_name = campaign_name or f"Campaign {results.campaign_id}"
        
        report = f"""# Creative Testing Report
## {campaign_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Summary

| Metric | Value |
|--------|-------|
| Total Tested | {results.total_creatives_tested} |
| Passed | {results.creatives_passed} |
| Failed | {results.creatives_failed} |
| Pass Rate | {results.pass_rate * 100:.0f}% |

---

## Creatives Tested

| Creative | Type | Lift | Stat Sig | Result |
|----------|------|------|----------|--------|
"""
        
        for r in results.results:
            result_str = "✓ PASS" if r.passed else "✗ FAIL"
            report += f"| {r.creative_name} | {r.asset_type.value} | {r.primary_kpi_lift:.1f}% | {'Yes' if r.primary_kpi_stat_sig else 'No'} | {result_str} |\n"
        
        report += """
---

## Recommendations

### Run These Creatives
"""
        if recommendations.run_creatives:
            for cid in recommendations.run_creatives:
                rec = next((r for r in recommendations.recommendations if r.creative_id == cid), None)
                if rec:
                    report += f"- **{rec.creative_name}**: {rec.rationale}\n"
        else:
            report += "- None passed testing\n"
        
        report += """
### Do Not Run
"""
        if recommendations.do_not_run_creatives:
            for cid in recommendations.do_not_run_creatives:
                rec = next((r for r in recommendations.recommendations if r.creative_id == cid), None)
                if rec:
                    report += f"- **{rec.creative_name}**: {rec.rationale}\n"
                    if rec.diagnostic_insights:
                        report += f"  - Issues: {'; '.join(rec.diagnostic_insights[:3])}\n"
                    if rec.suggested_improvements:
                        report += f"  - Suggested: {'; '.join(rec.suggested_improvements[:3])}\n"
        else:
            report += "- All creatives passed!\n"
        
        report += """
---

## Long-Term Recommendations
"""
        for rec in recommendations.long_term_recommendations:
            report += f"- {rec}\n"
        
        if recommendations.meta_insights:
            report += """
## Meta Insights
"""
            for insight in recommendations.meta_insights:
                report += f"- {insight}\n"
        
        return report

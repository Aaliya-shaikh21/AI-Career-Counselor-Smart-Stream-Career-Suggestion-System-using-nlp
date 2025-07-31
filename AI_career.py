import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time


class AICareerCounselor:
    def __init__(self, dataset_path=r"C:\Users\Aaliya Shaikh\PycharmProjects\PythonProject1\data\career_dataset.csv"):
        """Initialize the AI Career Counselor with dataset"""
        try:
            self.df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            # Create sample data if file not found
            self.df = self.create_sample_dataset()

        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.setup_data()

    def create_sample_dataset(self):
        """Create sample dataset if CSV file is not available"""
        sample_data = {
            'career_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'field': ['Software Engineer', 'Data Scientist', 'UX Designer', 'AI Researcher',
                      'Cybersecurity Analyst', 'Mobile Developer', 'Cloud Architect',
                      'Digital Marketer', 'Financial Analyst', 'Content Writer'],
            'description': [
                'Develop software applications and systems',
                'Extract insights from complex datasets',
                'Create intuitive digital experiences',
                'Develop cutting-edge AI models',
                'Secure computer systems and networks',
                'Build apps for iOS/Android platforms',
                'Design and manage cloud infrastructure',
                'Promote products/services online',
                'Analyze financial data and trends',
                'Create engaging written content'
            ],
            'skills': [
                'Python, Java, Algorithms, Debugging',
                'Python, SQL, Machine Learning, Statistics',
                'Figma, User Research, Wireframing, UI Principles',
                'Python, TensorFlow, Research Skills, Math',
                'Ethical Hacking, Network Security, Cryptography',
                'Swift, Kotlin, React Native, UI Design',
                'AWS, Azure, Docker, Kubernetes',
                'SEO, Social Media, Analytics',
                'Excel, Accounting, Market Research',
                'Writing Skills, Research, SEO'
            ],
            'interests': [
                'coding problem solving technology',
                'data analysis patterns statistics',
                'design user interface experience',
                'artificial intelligence machine learning',
                'security protection cyber attacks',
                'mobile applications development',
                'cloud computing infrastructure',
                'marketing consumer behavior',
                'finance investments analysis',
                'writing communication content'
            ],
            'salary': ['$80,000-$150,000', '$90,000-$160,000', '$70,000-$120,000',
                       '$100,000-$200,000', '$85,000-$140,000', '$80,000-$140,000',
                       '$110,000-$180,000', '$60,000-$110,000', '$70,000-$130,000', '$40,000-$80,000'],
            'growth': ['Very High', 'Very High', 'High', 'Very High', 'Very High',
                       'High', 'Very High', 'High', 'High', 'Moderate']
        }
        return pd.DataFrame(sample_data)

    def setup_data(self):
        """Preprocess and prepare the dataset"""
        # Clean salary data and extract numeric values
        self.df['salary_min'] = self.df['salary'].str.extract(r'\$(\d+),?(\d+)?').apply(
            lambda x: int(x[0] + (x[1] if pd.notna(x[1]) else '')) * 1000 if pd.notna(x[0]) else 50000, axis=1
        )

        # Encode growth levels
        growth_mapping = {'Very High': 5, 'High': 4, 'Moderate': 3, 'Low': 2, 'Very Low': 1}
        self.df['growth_score'] = self.df['growth'].map(growth_mapping).fillna(3)

        # Combine text features for similarity analysis
        self.df['combined_features'] = (
                self.df['interests'].fillna('') + ' ' +
                self.df['description'].fillna('') + ' ' +
                self.df['skills'].fillna('')
        )

        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'])

    def analyze_user_input(self, user_interests, preferred_skills=None, salary_preference=None):
        """Analyze user input and return career recommendations"""
        # Combine user input
        user_text = user_interests
        if preferred_skills:
            user_text += ' ' + preferred_skills

        # Transform user input using the same vectorizer
        user_vector = self.vectorizer.transform([user_text])

        # Calculate similarity scores
        similarity_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        # Create recommendations dataframe
        recommendations = self.df.copy()
        recommendations['similarity_score'] = similarity_scores

        # Apply filters
        if salary_preference and salary_preference > 0:
            recommendations = recommendations[
                recommendations['salary_min'] >= salary_preference
                ]

        # Sort by similarity score
        recommendations = recommendations.sort_values('similarity_score', ascending=False)

        return recommendations.head(5)

    def get_career_roadmap(self, career_field):
        """Generate a career roadmap for specific career"""
        career_row = self.df[self.df['field'] == career_field]
        if career_row.empty:
            return None

        career = career_row.iloc[0]

        roadmap = {
            'career': career['field'],
            'description': career['description'],
            'required_skills': career['skills'].split(', '),
            'salary_range': career['salary'],
            'growth_potential': career['growth'],
            'learning_path': self.generate_learning_path(career['skills']),
            'timeline': '6-12 months for entry level'
        }

        return roadmap

    def generate_learning_path(self, skills_str):
        """Generate learning recommendations based on required skills"""
        skills = skills_str.split(', ')
        learning_path = []

        skill_resources = {
            'Python': ['Learn Python basics', 'Practice with projects', 'Study data structures'],
            'Java': ['Master OOP concepts', 'Build applications', 'Learn frameworks'],
            'SQL': ['Database fundamentals', 'Query optimization', 'Database design'],
            'Machine Learning': ['Statistics basics', 'ML algorithms', 'Hands-on projects'],
            'Research': ['Academic writing', 'Literature review', 'Methodology'],
            'Communication': ['Public speaking', 'Technical writing', 'Presentation skills'],
            'Leadership': ['Team management', 'Project planning', 'Decision making'],
            'Design': ['Design principles', 'User research', 'Prototyping'],
            'Marketing': ['Digital marketing basics', 'Analytics', 'Campaign management']
        }

        for skill in skills:
            for key in skill_resources:
                if key.lower() in skill.lower():
                    learning_path.extend(skill_resources[key])
                    break
            else:
                learning_path.append(f'Study {skill} fundamentals')

        return learning_path[:6]  # Return top 6 recommendations

    def skill_gap_analysis(self, current_skills, target_career_field):
        """Analyze skill gaps for a target career"""
        target_career = self.df[self.df['field'] == target_career_field]
        if target_career.empty:
            return None

        required_skills = set(skill.strip() for skill in target_career.iloc[0]['skills'].split(','))
        current_skills_set = set(skill.strip() for skill in current_skills)

        skill_gaps = required_skills - current_skills_set
        matching_skills = required_skills & current_skills_set

        return {
            'matching_skills': list(matching_skills),
            'skill_gaps': list(skill_gaps),
            'match_percentage': len(matching_skills) / len(required_skills) * 100 if required_skills else 0
        }


class ModernCareerCounselorGUI:
    def __init__(self, root):
        self.root = root
        self.counselor = AICareerCounselor('career_dataset.csv')
        self.current_recommendations = None
        self.setup_gui()
        self.apply_modern_theme()

    def setup_gui(self):
        """Setup the main GUI structure"""
        self.root.title("🎓 AI Career Counselor - Smart Stream & Career Suggestion System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Create main container with paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Input
        self.create_input_panel(main_paned)

        # Right panel - Results
        self.create_results_panel(main_paned)

    def create_input_panel(self, parent):
        """Create input panel with modern styling"""
        input_frame = ttk.Frame(parent)
        parent.add(input_frame, weight=1)

        # Header
        header_frame = ttk.Frame(input_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        title_label = ttk.Label(header_frame, text="🎓 AI Career Assessment",
                                font=('Arial', 16, 'bold'))
        title_label.pack()

        subtitle_label = ttk.Label(header_frame, text="Smart Stream & Career Suggestion System",
                                   font=('Arial', 10))
        subtitle_label.pack()

        # Personal Information Section
        self.create_personal_info_section(input_frame)

        # Preferences Section
        self.create_preferences_section(input_frame)

        # Action Buttons
        self.create_action_buttons(input_frame)

    def create_personal_info_section(self, parent):
        """Create personal information input section"""
        info_frame = ttk.LabelFrame(parent, text="📝 Personal Information", padding=15)
        info_frame.pack(fill=tk.X, pady=(0, 15))

        # Interests
        ttk.Label(info_frame, text="What are your interests and passions?",
                  font=('Arial', 9, 'bold')).pack(anchor=tk.W)
        self.interests_text = scrolledtext.ScrolledText(info_frame, height=4, width=50,
                                                        font=('Arial', 9))
        self.interests_text.pack(fill=tk.X, pady=(5, 15))
        self.interests_text.insert(1.0, "e.g., I love coding, solving problems, working with data...")

        # Skills
        ttk.Label(info_frame, text="Current Skills (comma-separated):",
                  font=('Arial', 9, 'bold')).pack(anchor=tk.W)
        self.skills_entry = ttk.Entry(info_frame, width=50, font=('Arial', 9))
        self.skills_entry.pack(fill=tk.X, pady=(5, 15))
        self.skills_entry.insert(0, "e.g., Python, Communication, Problem Solving")

        # Salary preference
        salary_frame = ttk.Frame(info_frame)
        salary_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(salary_frame, text="Minimum Salary Preference ($):",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        self.salary_var = tk.StringVar(value="70000")
        salary_spinbox = ttk.Spinbox(salary_frame, from_=30000, to=300000,
                                     increment=5000, textvariable=self.salary_var,
                                     width=15, font=('Arial', 9))
        salary_spinbox.pack(side=tk.LEFT, padx=(10, 0))

    def create_preferences_section(self, parent):
        """Create preferences section"""
        pref_frame = ttk.LabelFrame(parent, text="⚙️ Analysis Options", padding=15)
        pref_frame.pack(fill=tk.X, pady=(0, 15))

        self.show_roadmap = tk.BooleanVar(value=True)
        self.show_skills = tk.BooleanVar(value=True)
        self.show_visualization = tk.BooleanVar(value=True)

        ttk.Checkbutton(pref_frame, text="Generate Career Roadmap",
                        variable=self.show_roadmap,
                        style='Modern.TCheckbutton').pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(pref_frame, text="Analyze Skill Gaps",
                        variable=self.show_skills,
                        style='Modern.TCheckbutton').pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(pref_frame, text="Show Visualizations",
                        variable=self.show_visualization,
                        style='Modern.TCheckbutton').pack(anchor=tk.W, pady=2)

    def create_action_buttons(self, parent):
        """Create action buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=15)

        # Main analyze button
        self.analyze_btn = ttk.Button(button_frame, text="🔍 Analyze Career Options",
                                      command=self.start_analysis,
                                      style='Accent.TButton')
        self.analyze_btn.pack(fill=tk.X, pady=(0, 10))

        # Progress bar
        self.progress = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 15))

        # Status label
        self.status_label = ttk.Label(button_frame, text="Ready to analyze your career options",
                                      font=('Arial', 8))
        self.status_label.pack()

        # Quick action buttons
        quick_frame = ttk.LabelFrame(parent, text="🚀 Quick Actions", padding=10)
        quick_frame.pack(fill=tk.X, pady=(15, 0))

        ttk.Button(quick_frame, text="📈 Market Trends",
                   command=self.show_trends,
                   style='Secondary.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="🧠 Personality Assessment",
                   command=self.personality_test,
                   style='Secondary.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="💡 Career Tips",
                   command=self.show_career_tips,
                   style='Secondary.TButton').pack(fill=tk.X, pady=2)

    def create_results_panel(self, parent):
        """Create results panel with notebook tabs"""
        results_frame = ttk.Frame(parent)
        parent.add(results_frame, weight=2)

        # Results header
        header_frame = ttk.Frame(results_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        self.results_header = ttk.Label(header_frame, text="🎯 Career Analysis Results",
                                        font=('Arial', 14, 'bold'))
        self.results_header.pack()

        self.results_subtitle = ttk.Label(header_frame, text="Discover your perfect career match",
                                          font=('Arial', 9))
        self.results_subtitle.pack()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.create_recommendations_tab()
        self.create_roadmap_tab()
        self.create_skills_tab()
        self.create_visualization_tab()

    def create_recommendations_tab(self):
        """Create recommendations tab"""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="🎯 Recommendations")

        # Create scrollable frame
        canvas = tk.Canvas(rec_frame)
        scrollbar = ttk.Scrollbar(rec_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Treeview for recommendations
        columns = ('Rank', 'Career', 'Match', 'Salary', 'Growth')
        self.recommendations_tree = ttk.Treeview(scrollable_frame, columns=columns,
                                                 show='headings', height=10)

        # Configure columns
        column_widths = {'Rank': 50, 'Career': 200, 'Match': 80, 'Salary': 150, 'Growth': 100}
        for col in columns:
            self.recommendations_tree.heading(col, text=col)
            self.recommendations_tree.column(col, width=column_widths.get(col, 100))

        self.recommendations_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Details frame
        details_frame = ttk.LabelFrame(scrollable_frame, text="Career Details", padding=10)
        details_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.career_details = scrolledtext.ScrolledText(details_frame, height=8)
        self.career_details.pack(fill=tk.BOTH, expand=True)

        # Bind selection event
        self.recommendations_tree.bind('<<TreeviewSelect>>', self.on_career_select)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_roadmap_tab(self):
        """Create roadmap tab"""
        roadmap_frame = ttk.Frame(self.notebook)
        self.notebook.add(roadmap_frame, text="🗺️ Career Roadmap")

        # Career selection
        selection_frame = ttk.Frame(roadmap_frame)
        selection_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(selection_frame, text="Select Career:",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.career_combo = ttk.Combobox(selection_frame, width=30, font=('Arial', 9))
        self.career_combo.pack(side=tk.LEFT, padx=(10, 5))

        ttk.Button(selection_frame, text="Generate Roadmap",
                   command=self.generate_roadmap,
                   style='Accent.TButton').pack(side=tk.LEFT, padx=(5, 0))

        # Roadmap display
        self.roadmap_text = scrolledtext.ScrolledText(roadmap_frame, font=('Arial', 9))
        self.roadmap_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def create_skills_tab(self):
        """Create skills analysis tab"""
        skills_frame = ttk.Frame(self.notebook)
        self.notebook.add(skills_frame, text="🛠️ Skills Analysis")

        # Skills gap analysis
        analysis_frame = ttk.LabelFrame(skills_frame, text="Skill Gap Analysis", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.skills_analysis = scrolledtext.ScrolledText(analysis_frame, font=('Arial', 9))
        self.skills_analysis.pack(fill=tk.BOTH, expand=True)

    def create_visualization_tab(self):
        """Create visualization tab"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="📊 Visualizations")

        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def start_analysis(self):
        """Start career analysis"""
        interests = self.interests_text.get(1.0, tk.END).strip()
        skills = self.skills_entry.get().strip()

        # Clear placeholder text
        if interests.startswith("e.g.,"):
            interests = ""
        if skills.startswith("e.g.,"):
            skills = ""

        if not interests:
            messagebox.showwarning("Input Required", "Please enter your interests and passions!")
            return

        # Show progress
        self.progress.start()
        self.analyze_btn.configure(state='disabled')
        self.status_label.configure(text="Analyzing your profile...")

        # Run analysis in thread to keep GUI responsive
        def run_analysis():
            try:
                salary_pref = int(self.salary_var.get()) if self.salary_var.get() else None
                recommendations = self.counselor.analyze_user_input(interests, skills, salary_pref)

                # Update GUI in main thread
                self.root.after(0, self.display_results, recommendations)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
                self.root.after(0, self.analysis_complete)

        threading.Thread(target=run_analysis, daemon=True).start()

    def display_results(self, recommendations):
        """Display analysis results"""
        self.current_recommendations = recommendations

        # Clear existing items
        for item in self.recommendations_tree.get_children():
            self.recommendations_tree.delete(item)

        # Add new recommendations
        for i, (_, row) in enumerate(recommendations.iterrows()):
            self.recommendations_tree.insert('', 'end', values=(
                i + 1,
                row['field'],
                f"{row['similarity_score']:.2f}",
                row['salary'],
                row['growth']
            ))

        # Update career combo
        career_list = list(recommendations['field'])
        self.career_combo['values'] = career_list
        if career_list:
            self.career_combo.set(career_list[0])

        # Create visualization if enabled
        if self.show_visualization.get():
            self.create_visualization(recommendations)

        # Generate roadmap for top career if enabled
        if self.show_roadmap.get() and career_list:
            self.generate_roadmap()

        # Analyze skills if enabled
        if self.show_skills.get() and career_list:
            self.analyze_skills()

        self.results_header.configure(text=f"✅ Found {len(recommendations)} Career Matches")
        self.results_subtitle.configure(text="Click on a career below to see detailed information")

        self.analysis_complete()

    def analysis_complete(self):
        """Complete analysis and reset UI"""
        self.progress.stop()
        self.analyze_btn.configure(state='normal')
        self.status_label.configure(text="Analysis complete! Check the results tabs.")

    def on_career_select(self, event):
        """Handle career selection in treeview"""
        selection = self.recommendations_tree.selection()
        if selection and self.current_recommendations is not None:
            item = self.recommendations_tree.item(selection[0])
            career_name = item['values'][1]

            # Find career details
            career_row = self.current_recommendations[
                self.current_recommendations['field'] == career_name
                ]

            if not career_row.empty:
                career = career_row.iloc[0]
                details = f"""🎯 {career['field']}

📝 Description:
{career['description']}

💰 Salary Range: {career['salary']}
📈 Growth Potential: {career['growth']}
🎯 Match Score: {career['similarity_score']:.2f}

🛠️ Required Skills:
{career['skills']}

💡 Key Interests:
{career['interests']}
"""
                self.career_details.delete(1.0, tk.END)
                self.career_details.insert(1.0, details)

    def generate_roadmap(self):
        """Generate career roadmap"""
        selected_career = self.career_combo.get()
        if selected_career:
            roadmap = self.counselor.get_career_roadmap(selected_career)

            if roadmap:
                roadmap_text = f"""🎯 CAREER ROADMAP: {roadmap['career']}

📝 DESCRIPTION:
{roadmap['description']}

💰 SALARY RANGE: {roadmap['salary_range']}
📈 GROWTH POTENTIAL: {roadmap['growth_potential']}

🛠️ REQUIRED SKILLS:
{chr(10).join(f"• {skill}" for skill in roadmap['required_skills'])}

📚 RECOMMENDED LEARNING PATH:
{chr(10).join(f"{i + 1}. {step}" for i, step in enumerate(roadmap['learning_path']))}

⏱️ ESTIMATED TIMELINE: {roadmap['timeline']}

💡 NEXT STEPS:
1. Start with foundational skills listed above
2. Build a portfolio showcasing your projects
3. Network with professionals in the field
4. Consider relevant certifications
5. Apply for entry-level positions or internships
"""
                self.roadmap_text.delete(1.0, tk.END)
                self.roadmap_text.insert(1.0, roadmap_text)
            else:
                self.roadmap_text.delete(1.0, tk.END)
                self.roadmap_text.insert(1.0, "❌ Could not generate roadmap for selected career.")

    def analyze_skills(self):
        """Analyze skill gaps"""
        selected_career = self.career_combo.get()
        current_skills_text = self.skills_entry.get().strip()

        if selected_career and current_skills_text:
            # Remove placeholder text
            if current_skills_text.startswith("e.g.,"):
                current_skills_text = ""

            current_skills = [skill.strip() for skill in current_skills_text.split(',') if skill.strip()]
            gap_analysis = self.counselor.skill_gap_analysis(current_skills, selected_career)

            if gap_analysis:
                analysis_text = f"""🎯 SKILLS ANALYSIS FOR: {selected_career}

📊 OVERALL MATCH: {gap_analysis['match_percentage']:.1f}%

✅ SKILLS YOU ALREADY HAVE:
{chr(10).join(f"• {skill}" for skill in gap_analysis['matching_skills']) if gap_analysis['matching_skills'] else "• None identified"}

❌ SKILLS YOU NEED TO DEVELOP:
{chr(10).join(f"• {skill}" for skill in gap_analysis['skill_gaps']) if gap_analysis['skill_gaps'] else "• You have all required skills!"}

💡 RECOMMENDATIONS:
"""
                if gap_analysis['skill_gaps']:
                    analysis_text += "Focus on developing the missing skills through:\n"
                    analysis_text += "• Online courses (Coursera, Udemy, edX)\n"
                    analysis_text += "• Hands-on projects and practice\n"
                    analysis_text += "• Professional certifications\n"
                    analysis_text += "• Mentorship and networking\n"
                else:
                    analysis_text += "Great! You have all the required skills. Consider:\n"
                    analysis_text += "• Building a strong portfolio\n"
                    analysis_text += "• Gaining practical experience\n"
                    analysis_text += "• Developing advanced expertise\n"

                self.skills_analysis.delete(1.0, tk.END)
                self.skills_analysis.insert(1.0, analysis_text)
            else:
                self.skills_analysis.delete(1.0, tk.END)
                self.skills_analysis.insert(1.0, "❌ Could not analyze skills for selected career.")

    def create_visualization(self, recommendations):
        """Create visualization of recommendations"""
        self.figure.clear()

        # Create subplots
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(212)

        # Data for charts
        careers = recommendations['field'].head(5)
        scores = recommendations['similarity_score'].head(5)
        growth_scores = recommendations['growth_score'].head(5)
        salary_mins = recommendations['salary_min'].head(5)

        # Chart 1: Match Scores
        bars1 = ax1.barh(careers, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        ax1.set_xlabel('Match Score')
        ax1.set_title('Career Match Scores', fontweight='bold')
        ax1.set_xlim(0, 1)

        # Add value labels
        for bar, score in zip(bars1, scores):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{score:.2f}', va='center', fontweight='bold')

        # Chart 2: Growth Potential
        bars2 = ax2.bar(range(len(careers)), growth_scores,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        ax2.set_xlabel('Careers')
        ax2.set_ylabel('Growth Score')
        ax2.set_title('Growth Potential', fontweight='bold')
        ax2.set_xticks(range(len(careers)))
        ax2.set_xticklabels([career[:10] + '...' if len(career) > 10 else career
                             for career in careers], rotation=45)

        # Chart 3: Salary Comparison
        bars3 = ax3.bar(careers, salary_mins / 1000,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        ax3.set_ylabel('Minimum Salary (K$)')
        ax3.set_title('Salary Comparison', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, salary in zip(bars3, salary_mins):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'${salary // 1000}K', ha='center', fontweight='bold')

        self.figure.tight_layout()
        self.canvas.draw()

    def show_trends(self):
        """Show market trends"""
        trends_info = """📈 CURRENT MARKET TRENDS (2025)

🔥 HIGH GROWTH FIELDS:
• Artificial Intelligence & Machine Learning (35% growth)
• Cybersecurity (28% growth)
• Cloud Computing (30% growth)
• Data Science & Analytics (25% growth)
• Mobile & Web Development (20% growth)

💼 EMERGING OPPORTUNITIES:
• AI Ethics & Governance
• Quantum Computing
• Sustainable Technology
• Virtual/Augmented Reality
• Blockchain & Web3

📊 INDUSTRY INSIGHTS:
• Remote work capabilities are increasingly valued
• Cross-functional skills are in high demand
• Continuous learning is essential for career growth
• Soft skills complement technical expertise
• Automation is creating new job categories"""

        messagebox.showinfo("Market Trends 2025", trends_info)

    def personality_test(self):
        """Start personality assessment"""
        messagebox.showinfo("Personality Assessment",
                            "🧠 Advanced personality assessment coming soon!\n\n"
                            "This feature will include:\n"
                            "• Myers-Briggs Type Indicator\n"
                            "• Career interest inventory\n"
                            "• Work style preferences\n"
                            "• Leadership potential analysis")

    def show_career_tips(self):
        """Show career development tips"""
        tips_info = """💡 CAREER DEVELOPMENT TIPS

🎯 CAREER PLANNING:
• Set clear short-term and long-term goals
• Research industry trends and requirements
• Build a professional network
• Seek mentorship opportunities

🛠️ SKILL DEVELOPMENT:
• Focus on both technical and soft skills
• Take online courses and certifications
• Work on real-world projects
• Practice continuous learning

📝 JOB SEARCH STRATEGY:
• Tailor your resume for each application
• Build a strong online presence (LinkedIn)
• Prepare for behavioral interviews
• Follow up professionally

🚀 CAREER ADVANCEMENT:
• Take on challenging projects
• Seek feedback and act on it
• Document your achievements
• Consider internal mobility opportunities"""

        messagebox.showinfo("Career Development Tips", tips_info)

    def apply_modern_theme(self):
        """Apply modern theme to the interface"""
        style = ttk.Style()

        # Use a modern theme
        style.theme_use('clam')

        # Configure custom styles
        style.configure('Accent.TButton',
                        foreground='white',
                        background='#4CAF50',
                        font=('Arial', 10, 'bold'))
        style.map('Accent.TButton',
                  background=[('active', '#45a049')])

        style.configure('Secondary.TButton',
                        foreground='#333333',
                        background='#e0e0e0',
                        font=('Arial', 9))
        style.map('Secondary.TButton',
                  background=[('active', '#d0d0d0')])

        style.configure('Modern.TCheckbutton',
                        font=('Arial', 9))

        # Configure other widgets
        style.configure('TLabelframe',
                        foreground='#333333',
                        font=('Arial', 10, 'bold'))
        style.configure('TLabelframe.Label',
                        foreground='#333333',
                        font=('Arial', 10, 'bold'))


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = ModernCareerCounselorGUI(root)

    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1200 // 2)
    y = (root.winfo_screenheight() // 2) - (800 // 2)
    root.geometry(f"1200x800+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()

# Contributing to PulmoVision AI

Thank you for your interest in contributing to PulmoVision AI! Contributions from the community help make this tool better for everyone.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

- Search existing issues to see if the bug has already been reported.
- If not, open a new issue using the **Bug report** template.
- Include a clear title, description, steps to reproduce, and any relevant error logs or screenshots.

### Suggesting Enhancements

- Open an issue using the **Feature request** template.
- Explain the feature, use cases, and how it benefits the project.

### Pull Requests

1. **Fork the repository** and create your branch from `main`.
2. **Install dependencies** in a virtual environment:
   ```bash
   python -m venv venv310
   # Windows:
   venv310\Scripts\activate
   # Linux/macOS:
   source venv310/bin/activate

   pip install -r requirements.txt
   pip install pytest flake8 black
   ```
3. **Make your changes**. Ensure your code is clean and follows standard PEP 8 format.
4. **Write tests** for your new code in the `tests/` directory.
5. **Run tests** and verify everything passes:
   ```bash
   python -m pytest tests/
   ```
6. **Submit a pull request** matching the Pull Request Template.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

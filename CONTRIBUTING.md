# Contributing to Security Camera System

Thank you for your interest in contributing to the Security Camera System! This document provides guidelines and steps for contributing to this project.

## Development Environment Setup

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/security-script.git
cd security-script
```

2. Set up Python environment using pyenv:
```bash
pyenv local 3.11.0
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Code Style Guidelines

1. Follow PEP 8 style guide for Python code
2. Use type hints for function parameters and return values
3. Include docstrings for classes and functions
4. Keep functions focused and single-purpose
5. Use meaningful variable and function names

## Pull Request Process

1. Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

2. Make your changes, following our code style guidelines

3. Add or update tests as needed

4. Commit your changes with clear, descriptive commit messages:
```bash
git commit -m "feat: Add face detection timeout feature"
# or
git commit -m "fix: Resolve camera initialization error"
```

5. If you have multiple commits, compress them into a single, clean commit:
```bash
# Assuming you're on your feature branch and want to compress the last n commits
git rebase -i HEAD~n

# In the editor, mark commits to squash by changing 'pick' to 'squash' or 's'
# Leave the first commit as 'pick'
# Save and exit the editor
# In the next editor, write a clear commit message for the compressed commit
```

6. Push to your fork and create a Pull Request

## Commit Message Guidelines

Follow the Conventional Commits specification:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for code style changes (formatting, missing semi-colons, etc)
- `refactor:` for code refactoring
- `test:` for adding or modifying tests
- `chore:` for routine tasks, maintenance, and dependency updates

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting a PR
- Include both positive and negative test cases
- Test edge cases and error conditions

## Reporting Issues

When reporting issues, please include:

1. Description of the issue
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. System information:
   - Operating System
   - Python version
   - Package versions (output of `pip freeze`)
   - Camera hardware details (if relevant)

## Code Review Process

1. All submissions require review
2. Changes must be tested on Linux
3. Reviewers will check for:
   - Code style compliance
   - Test coverage
   - Documentation completeness
   - Security implications
   - Performance impact

## Security Considerations

When contributing, please ensure:

1. No sensitive information is committed (API keys, credentials)
2. Input validation is properly implemented
3. Error handling follows security best practices
4. User data privacy is maintained
5. System resources are properly managed

## Documentation

When adding or modifying features:

1. Update relevant README sections
2. Add inline code comments for complex logic
3. Update function/class docstrings
4. Include usage examples if applicable

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions or Need Help?

Feel free to open an issue for:
- Feature discussions
- Implementation questions
- Clarification on contributing guidelines
- Help with development setup

Thank you for contributing to making this security camera system better!

# Mapping Notes: from_pages Table

## Overview
All columns in the `from_pages` table were sourced directly from the `AppObjects` table during initial import.

## General Rules
- All fields use **direct copy**, no joins needed
- JSON-type fields (`AppObjectConfiguration`, `Config`) are stored as stringified JSON and cleaned in Python
- Field `JsonSchema` has been deprecated and removed
- Data type conversion and cleaning were done via Python scripts

## Cleaning Rules Applied
- Trimmed all string fields
- Filled missing values with defaults
- Removed duplicate `Id` values
- Validated JSON structure in config-related fields

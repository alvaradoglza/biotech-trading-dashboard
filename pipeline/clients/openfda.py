"""OpenFDA API client for drug approvals and labels."""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

from pipeline.clients.base import BaseAPIClient, APIError
from pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class OpenFDAAPIError(APIError):
    """OpenFDA API-specific error."""

    pass


@dataclass
class DrugApproval:
    """Represents a drug approval from OpenFDA Drugs@FDA."""

    application_number: str
    sponsor_name: str
    brand_name: str
    generic_name: Optional[str] = None
    approval_date: Optional[date] = None
    submission_type: str = ""  # NDA, BLA, ANDA
    submission_status: str = ""
    dosage_form: str = ""
    route: str = ""
    active_ingredients: list[str] = field(default_factory=list)
    products: list[dict] = field(default_factory=list)

    @property
    def url(self) -> str:
        """Get the Drugs@FDA URL for this application."""
        return f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={self.application_number.replace('NDA', '').replace('BLA', '').replace('ANDA', '')}"


@dataclass
class DrugLabel:
    """Represents a drug label from OpenFDA."""

    application_number: str
    brand_name: str
    generic_name: Optional[str] = None
    manufacturer: Optional[str] = None
    indications_and_usage: Optional[str] = None
    warnings: Optional[str] = None
    dosage_and_administration: Optional[str] = None
    effective_date: Optional[date] = None


@dataclass
class DrugRecall:
    """Represents a drug recall/enforcement action from OpenFDA."""

    recall_number: str
    recalling_firm: str
    product_description: str
    reason_for_recall: str
    classification: str  # Class I, II, or III
    status: str  # Ongoing, Terminated, etc.
    recall_initiation_date: Optional[date] = None
    report_date: Optional[date] = None
    termination_date: Optional[date] = None
    voluntary_mandated: str = ""
    distribution_pattern: str = ""
    city: str = ""
    state: str = ""
    country: str = ""

    @property
    def url(self) -> str:
        """Get the FDA enforcement URL."""
        return f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm?id={self.recall_number}"


class OpenFDAClient(BaseAPIClient):
    """Client for OpenFDA API.

    The OpenFDA API provides access to:
    - Drugs@FDA: Drug application information
    - Drug labels: Package insert information
    - Adverse events: Drug adverse event reports

    API documentation: https://open.fda.gov/apis/

    Rate limits:
    - Without API key: 40 requests per minute, 1000 per day
    - With API key: 240 requests per minute
    """

    BASE_URL = "https://api.fda.gov"
    RATE_LIMIT_NO_KEY = 40  # per minute = 0.67/sec
    RATE_LIMIT_WITH_KEY = 240  # per minute = 4/sec

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """Initialize the OpenFDA client.

        Args:
            api_key: Optional API key for higher rate limits
        """
        self.has_api_key = bool(api_key)

        # Calculate requests per second based on rate limit
        if api_key:
            # 240 per minute = 4 per second (be conservative)
            requests_per_second = 3.0
        else:
            # 40 per minute = 0.67 per second (be conservative)
            requests_per_second = 0.5

        super().__init__(
            base_url=self.BASE_URL,
            api_key=api_key,
            requests_per_second=requests_per_second,
            user_agent="BiopharmaMonitor/1.0",
        )

    def _build_params(
        self,
        search: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> dict[str, Any]:
        """Build query parameters for OpenFDA API.

        Args:
            search: Search query string
            limit: Number of results to return
            skip: Number of results to skip

        Returns:
            Dictionary of query parameters
        """
        params: dict[str, Any] = {
            "limit": min(limit, 1000),
        }

        if skip > 0:
            params["skip"] = skip

        if search:
            params["search"] = search

        if self.api_key:
            params["api_key"] = self.api_key

        return params

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[date]:
        """Parse date string from OpenFDA response.

        Args:
            date_str: Date string in YYYYMMDD format

        Returns:
            Parsed date or None
        """
        if not date_str:
            return None

        # OpenFDA uses YYYYMMDD format
        try:
            return datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            # Try other formats
            formats = ["%Y-%m-%d", "%m/%d/%Y"]
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue

        logger.debug("Could not parse date", date_str=date_str)
        return None

    def _parse_drug_approval(self, result: dict[str, Any]) -> DrugApproval:
        """Parse a drug approval from API response.

        Args:
            result: Single result from API response

        Returns:
            DrugApproval object
        """
        # Application info
        application_number = result.get("application_number", "")
        sponsor_name = result.get("sponsor_name", "")

        # Get submission info from first submission or submissions array
        submissions = result.get("submissions", [])
        submission_type = ""
        submission_status = ""
        approval_date = None

        if submissions:
            # Get most recent submission
            latest = submissions[0]
            submission_type = latest.get("submission_type", "")
            submission_status = latest.get("submission_status", "")
            approval_date = self._parse_date(latest.get("submission_status_date"))

        # Product info
        products = result.get("products", [])
        brand_name = ""
        generic_name = ""
        dosage_form = ""
        route = ""
        active_ingredients = []

        if products:
            first_product = products[0]
            brand_name = first_product.get("brand_name", "")
            dosage_form = first_product.get("dosage_form", "")
            route = first_product.get("route", "")

            # Active ingredients
            ingredients = first_product.get("active_ingredients", [])
            for ingredient in ingredients:
                name = ingredient.get("name", "")
                if name:
                    active_ingredients.append(name)

        # Get generic name from openfda section if available
        openfda = result.get("openfda", {})
        if openfda:
            generic_names = openfda.get("generic_name", [])
            if generic_names:
                generic_name = generic_names[0]
            if not brand_name:
                brand_names = openfda.get("brand_name", [])
                if brand_names:
                    brand_name = brand_names[0]

        return DrugApproval(
            application_number=application_number,
            sponsor_name=sponsor_name,
            brand_name=brand_name,
            generic_name=generic_name,
            approval_date=approval_date,
            submission_type=submission_type,
            submission_status=submission_status,
            dosage_form=dosage_form,
            route=route,
            active_ingredients=active_ingredients,
            products=products,
        )

    async def get_drug_approvals(
        self,
        sponsor_name: Optional[str] = None,
        application_number: Optional[str] = None,
        brand_name: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> tuple[list[DrugApproval], int]:
        """Search for drug approvals.

        Args:
            sponsor_name: Sponsor/company name
            application_number: NDA/BLA/ANDA number
            brand_name: Brand name of drug
            limit: Number of results (max 1000)
            skip: Number of results to skip

        Returns:
            Tuple of (list of approvals, total count)
        """
        # Build search query
        search_parts = []

        if sponsor_name:
            # Escape special characters and use quotes for exact match
            escaped = sponsor_name.replace('"', '\\"')
            search_parts.append(f'sponsor_name:"{escaped}"')

        if application_number:
            search_parts.append(f"application_number:{application_number}")

        if brand_name:
            escaped = brand_name.replace('"', '\\"')
            search_parts.append(f'products.brand_name:"{escaped}"')

        search = " AND ".join(search_parts) if search_parts else None

        params = self._build_params(search=search, limit=limit, skip=skip)

        logger.debug("Searching drug approvals", params=params)

        try:
            response = await self.get_json("drug/drugsfda.json", params=params)
        except APIError as e:
            if e.status_code == 404:
                # No results found
                return [], 0
            raise OpenFDAAPIError(str(e), status_code=e.status_code) from e

        results = response.get("results", [])
        meta = response.get("meta", {})
        total = meta.get("results", {}).get("total", 0)

        approvals = [self._parse_drug_approval(r) for r in results]

        logger.info(
            "Found drug approvals",
            count=len(approvals),
            total=total,
        )

        return approvals, total

    async def get_drug_approvals_raw(
        self,
        sponsor_name: Optional[str] = None,
        application_number: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> tuple[list[tuple["DrugApproval", dict]], int]:
        """Search for drug approvals and return both parsed objects and raw JSON.

        Args:
            sponsor_name: Sponsor/company name
            application_number: NDA/BLA/ANDA number
            limit: Number of results (max 1000)
            skip: Number of results to skip

        Returns:
            Tuple of (list of (approval, raw_dict) tuples, total count)
        """
        search_parts = []

        if sponsor_name:
            escaped = sponsor_name.replace('"', '\\"')
            search_parts.append(f'sponsor_name:"{escaped}"')

        if application_number:
            search_parts.append(f"application_number:{application_number}")

        search = " AND ".join(search_parts) if search_parts else None
        params = self._build_params(search=search, limit=limit, skip=skip)

        try:
            response = await self.get_json("drug/drugsfda.json", params=params)
        except APIError as e:
            if e.status_code == 404:
                return [], 0
            raise OpenFDAAPIError(str(e), status_code=e.status_code) from e

        results = response.get("results", [])
        meta = response.get("meta", {})
        total = meta.get("results", {}).get("total", 0)

        approvals_with_raw = [(self._parse_drug_approval(r), r) for r in results]

        return approvals_with_raw, total

    async def get_approvals_by_sponsor(
        self,
        sponsor_name: str,
        days_back: int = 365,
    ) -> list[DrugApproval]:
        """Get all approvals for a sponsor within a time period.

        Args:
            sponsor_name: Company/sponsor name
            days_back: Number of days to look back

        Returns:
            List of approvals
        """
        all_approvals = []
        skip = 0
        limit = 100

        while True:
            approvals, total = await self.get_drug_approvals(
                sponsor_name=sponsor_name,
                limit=limit,
                skip=skip,
            )

            if not approvals:
                break

            # Filter by date
            cutoff_date = date.today() - timedelta(days=days_back)
            for approval in approvals:
                if approval.approval_date and approval.approval_date >= cutoff_date:
                    all_approvals.append(approval)
                elif not approval.approval_date:
                    # Include approvals without date
                    all_approvals.append(approval)

            skip += limit
            if skip >= total or len(approvals) < limit:
                break

        return all_approvals

    async def get_recent_approvals(
        self,
        days_back: int = 30,
    ) -> list[DrugApproval]:
        """Get recent drug approvals across all sponsors.

        Args:
            days_back: Number of days to look back

        Returns:
            List of recent approvals
        """
        # Build date range search - use proper syntax without +TO+
        cutoff_date = date.today() - timedelta(days=days_back)
        today = date.today()

        # OpenFDA uses [YYYYMMDD TO YYYYMMDD] format (no + signs)
        search = (
            f"submissions.submission_status_date:"
            f"[{cutoff_date.strftime('%Y%m%d')} TO {today.strftime('%Y%m%d')}]"
        )

        params = self._build_params(search=search, limit=100)

        logger.debug("Fetching recent approvals", params=params)

        try:
            response = await self.get_json("drug/drugsfda.json", params=params)
        except APIError as e:
            if e.status_code == 404:
                return []
            raise OpenFDAAPIError(str(e), status_code=e.status_code) from e

        results = response.get("results", [])
        return [self._parse_drug_approval(r) for r in results]

    async def get_drug_label(
        self,
        application_number: Optional[str] = None,
        brand_name: Optional[str] = None,
    ) -> Optional[DrugLabel]:
        """Get drug label information.

        Args:
            application_number: NDA/BLA number
            brand_name: Brand name of drug

        Returns:
            DrugLabel or None if not found
        """
        search_parts = []

        if application_number:
            # Clean up application number
            app_num = application_number.replace("NDA", "").replace("BLA", "").replace("ANDA", "")
            search_parts.append(f"openfda.application_number:{app_num}")

        if brand_name:
            escaped = brand_name.replace('"', '\\"')
            search_parts.append(f'openfda.brand_name:"{escaped}"')

        if not search_parts:
            return None

        search = " AND ".join(search_parts)
        params = self._build_params(search=search, limit=1)

        logger.debug("Fetching drug label", params=params)

        try:
            response = await self.get_json("drug/label.json", params=params)
        except APIError as e:
            if e.status_code == 404:
                return None
            raise OpenFDAAPIError(str(e), status_code=e.status_code) from e

        results = response.get("results", [])
        if not results:
            return None

        result = results[0]
        openfda = result.get("openfda", {})

        return DrugLabel(
            application_number=application_number or "",
            brand_name=openfda.get("brand_name", [""])[0] if openfda.get("brand_name") else "",
            generic_name=openfda.get("generic_name", [""])[0] if openfda.get("generic_name") else None,
            manufacturer=openfda.get("manufacturer_name", [""])[0] if openfda.get("manufacturer_name") else None,
            indications_and_usage=result.get("indications_and_usage", [""])[0] if result.get("indications_and_usage") else None,
            warnings=result.get("warnings", [""])[0] if result.get("warnings") else None,
            dosage_and_administration=result.get("dosage_and_administration", [""])[0] if result.get("dosage_and_administration") else None,
            effective_date=self._parse_date(result.get("effective_time")),
        )

    async def search_approvals_by_ingredient(
        self,
        ingredient_name: str,
        limit: int = 100,
    ) -> list[DrugApproval]:
        """Search for drug approvals by active ingredient.

        Args:
            ingredient_name: Active ingredient name
            limit: Maximum results

        Returns:
            List of approvals containing the ingredient
        """
        escaped = ingredient_name.replace('"', '\\"')
        search = f'products.active_ingredients.name:"{escaped}"'

        params = self._build_params(search=search, limit=limit)

        try:
            response = await self.get_json("drug/drugsfda.json", params=params)
        except APIError as e:
            if e.status_code == 404:
                return []
            raise OpenFDAAPIError(str(e), status_code=e.status_code) from e

        results = response.get("results", [])
        return [self._parse_drug_approval(r) for r in results]

    def _parse_drug_recall(self, result: dict[str, Any]) -> DrugRecall:
        """Parse a drug recall from API response.

        Args:
            result: Single result from API response

        Returns:
            DrugRecall object
        """
        return DrugRecall(
            recall_number=result.get("recall_number", ""),
            recalling_firm=result.get("recalling_firm", ""),
            product_description=result.get("product_description", ""),
            reason_for_recall=result.get("reason_for_recall", ""),
            classification=result.get("classification", ""),
            status=result.get("status", ""),
            recall_initiation_date=self._parse_date(result.get("recall_initiation_date")),
            report_date=self._parse_date(result.get("report_date")),
            termination_date=self._parse_date(result.get("termination_date")),
            voluntary_mandated=result.get("voluntary_mandated", ""),
            distribution_pattern=result.get("distribution_pattern", ""),
            city=result.get("city", ""),
            state=result.get("state", ""),
            country=result.get("country", ""),
        )

    async def get_drug_recalls(
        self,
        recalling_firm: Optional[str] = None,
        classification: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> tuple[list[DrugRecall], int]:
        """Search for drug recalls/enforcement actions.

        Args:
            recalling_firm: Company name
            classification: Class I, Class II, or Class III
            status: Ongoing, Terminated, etc.
            limit: Number of results (max 1000)
            skip: Number of results to skip

        Returns:
            Tuple of (list of recalls, total count)
        """
        search_parts = []

        if recalling_firm:
            # Use wildcard search for better matching
            escaped = recalling_firm.replace('"', '\\"')
            search_parts.append(f'recalling_firm:"{escaped}"')

        if classification:
            search_parts.append(f'classification:"{classification}"')

        if status:
            search_parts.append(f'status:"{status}"')

        search = " AND ".join(search_parts) if search_parts else None

        params = self._build_params(search=search, limit=limit, skip=skip)

        logger.debug("Searching drug recalls", params=params)

        try:
            response = await self.get_json("drug/enforcement.json", params=params)
        except APIError as e:
            if e.status_code == 404:
                return [], 0
            raise OpenFDAAPIError(str(e), status_code=e.status_code) from e

        results = response.get("results", [])
        meta = response.get("meta", {})
        total = meta.get("results", {}).get("total", 0)

        recalls = [self._parse_drug_recall(r) for r in results]

        logger.info(
            "Found drug recalls",
            count=len(recalls),
            total=total,
        )

        return recalls, total

    async def get_recalls_by_firm(
        self,
        firm_name: str,
        days_back: int = 365,
    ) -> list[DrugRecall]:
        """Get all recalls for a firm within a time period.

        Args:
            firm_name: Company/firm name
            days_back: Number of days to look back

        Returns:
            List of recalls
        """
        all_recalls = []
        skip = 0
        limit = 100

        # Try variations of the firm name
        search_names = [
            firm_name,
            firm_name.split(",")[0].strip(),
            firm_name.replace(" Inc.", "").replace(" Corp.", "").replace(" Inc", "").replace(" Corp", "").strip(),
        ]

        # Remove duplicates
        seen = set()
        unique_names = []
        for name in search_names:
            if name and name not in seen:
                seen.add(name)
                unique_names.append(name)

        for search_name in unique_names:
            skip = 0
            while True:
                recalls, total = await self.get_drug_recalls(
                    recalling_firm=search_name,
                    limit=limit,
                    skip=skip,
                )

                if not recalls:
                    break

                # Filter by date
                cutoff_date = date.today() - timedelta(days=days_back)
                for recall in recalls:
                    recall_date = recall.recall_initiation_date or recall.report_date
                    if recall_date and recall_date >= cutoff_date:
                        # Avoid duplicates by recall number
                        if not any(r.recall_number == recall.recall_number for r in all_recalls):
                            all_recalls.append(recall)

                skip += limit
                if skip >= total or len(recalls) < limit:
                    break

            if all_recalls:
                # Found recalls with this name variation, stop trying others
                break

        return all_recalls

    async def get_recent_recalls(
        self,
        days_back: int = 30,
        classification: Optional[str] = None,
    ) -> list[DrugRecall]:
        """Get recent drug recalls across all firms.

        Args:
            days_back: Number of days to look back
            classification: Optional filter for Class I, II, or III

        Returns:
            List of recent recalls
        """
        cutoff_date = date.today() - timedelta(days=days_back)
        today = date.today()

        # Build search query
        search = (
            f"report_date:"
            f"[{cutoff_date.strftime('%Y%m%d')} TO {today.strftime('%Y%m%d')}]"
        )

        if classification:
            search += f' AND classification:"{classification}"'

        params = self._build_params(search=search, limit=100)

        logger.debug("Fetching recent recalls", params=params)

        try:
            response = await self.get_json("drug/enforcement.json", params=params)
        except APIError as e:
            if e.status_code == 404:
                return []
            raise OpenFDAAPIError(str(e), status_code=e.status_code) from e

        results = response.get("results", [])
        return [self._parse_drug_recall(r) for r in results]

    async def get_recalls_raw_by_firm(
        self,
        firm_name: str,
        days_back: int = 1095,
    ) -> list[tuple["DrugRecall", dict]]:
        """Get recalls for a firm within a time period, returning both parsed and raw dicts.

        Args:
            firm_name: Company/firm name
            days_back: Number of days to look back

        Returns:
            List of (DrugRecall, raw_dict) tuples
        """
        cutoff_date = date.today() - timedelta(days=days_back)

        search_names = [
            firm_name,
            firm_name.split(",")[0].strip(),
            firm_name.replace(" Inc.", "").replace(" Corp.", "").replace(" Inc", "").replace(" Corp", "").strip(),
        ]
        seen_names: set[str] = set()
        unique_names: list[str] = []
        for name in search_names:
            if name and name not in seen_names:
                seen_names.add(name)
                unique_names.append(name)

        all_results: list[tuple[DrugRecall, dict]] = []
        seen_recall_numbers: set[str] = set()

        for search_name in unique_names:
            skip = 0
            limit = 100
            while True:
                escaped = search_name.replace('"', '\\"')
                search = f'recalling_firm:"{escaped}"'
                params = self._build_params(search=search, limit=limit, skip=skip)

                try:
                    response = await self.get_json("drug/enforcement.json", params=params)
                except APIError as e:
                    if e.status_code == 404:
                        break
                    raise OpenFDAAPIError(str(e), status_code=e.status_code) from e

                raw_results = response.get("results", [])
                total = response.get("meta", {}).get("results", {}).get("total", 0)

                for raw in raw_results:
                    recall = self._parse_drug_recall(raw)
                    recall_date = recall.recall_initiation_date or recall.report_date
                    if recall_date and recall_date < cutoff_date:
                        continue
                    if recall.recall_number in seen_recall_numbers:
                        continue
                    seen_recall_numbers.add(recall.recall_number)
                    all_results.append((recall, raw))

                skip += limit
                if skip >= total or len(raw_results) < limit:
                    break

            if all_results:
                break  # Found results with this name variation

        return all_results

    async def get_all_fda_events_by_firm(
        self,
        firm_name: str,
        days_back: int = 365,
    ) -> dict[str, list]:
        """Get all FDA events (approvals + recalls) for a firm.

        This is the main method for fetching all FDA announcements
        for a company.

        Args:
            firm_name: Company/firm name
            days_back: Number of days to look back

        Returns:
            Dict with 'approvals' and 'recalls' lists
        """
        approvals = await self.get_approvals_by_sponsor(firm_name, days_back)
        recalls = await self.get_recalls_by_firm(firm_name, days_back)

        return {
            "approvals": approvals,
            "recalls": recalls,
        }
